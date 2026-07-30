// DepthToSpace: [N,C,H,W] -> [N, C/(b*b), H*b, W*b]. Rearranges blocks of depth into spatial.
// modes (ONNX): DCR (depth-column-row, default) and CRD (column-row-depth). NCHW fp32 reference.
//   out[n, c, h, w], with ih=h/b, bh=h%b, iw=w/b, bw=w%b, C2 = C/(b*b):
//     DCR: in channel = (bh*b + bw) * C2 + c
//     CRD: in channel = c * (b*b) + (bh*b + bw)
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        struct DepthToSpaceCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X = ctx.t(node.inputs[0]);
                RtTensor       &Y = ctx.t(node.outputs[0]);
                NCHW            x = NCHW::from(X.shape);
                int64_t         b = node.attr.geti("blocksize", 1);
                // blocksize < 1 is degenerate (would zero or invert the output spatial size); clamp to
                // the identity block so the op stays a well-defined copy rather than dividing by zero
                // in C2 = C/(b*b) below.
                if (b < 1)
                {
                    b = 1;
                }
                bool crd = node.attr.gets("mode", "DCR") == "CRD";
                // Output geometry: b*b input channels collapse into each output channel while H and W
                // each grow by b. C2 = C/(b*b) is exact because a valid DepthToSpace requires b*b | C.
                int64_t      C2 = x.c / (b * b), OH = x.h * b, OW = x.w * b;
                Shape        outShape = {x.n, C2, OH, OW};
                float       *y        = cpu::allocOut(Y, outShape);
                const float *in       = X.host.f32();
                int64_t      inHW     = x.h * x.w;
                // Drive the loop over the OUTPUT tensor and gather each element from its single source
                // in X, so every output position is written exactly once.
                for (int64_t n = 0; n < x.n; ++n)
                {
                    for (int64_t c = 0; c < C2; ++c)
                    {
                        for (int64_t h = 0; h < OH; ++h)
                        {
                            // Split the output row into the input row ih and the intra-block row bh
                            // (0..b-1); the block coordinates select which of the b*b depth planes fed
                            // this pixel.
                            int64_t ih = h / b, bh = h % b;
                            for (int64_t w = 0; w < OW; ++w)
                            {
                                int64_t iw = w / b, bw = w % b;
                                // blk in [0, b*b) is the row-major index of the (bh,bw) cell within the
                                // b x b block. DCR (default) lays block index ABOVE the output channel
                                // in the input-channel axis (channel = blk*C2 + c); CRD lays it BELOW
                                // (channel = c*b*b + blk). This is the only difference between the modes.
                                int64_t blk = bh * b + bw;
                                int64_t ic  = crd ? (c * (b * b) + blk) : (blk * C2 + c);
                                // Row-major (NCHW) flattening of the resolved 4-D coordinates: input at
                                // (n, ic, ih, iw) with X's original channel count x.c, output at
                                // (n, c, h, w) with C2 channels.
                                int64_t inIdx  = ((n * x.c + ic) * x.h + ih) * x.w + iw;
                                int64_t outIdx = ((n * C2 + c) * OH + h) * OW + w;
                                y[outIdx]      = in[inIdx];
                                (void) inHW;
                            }
                        }
                    }
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::DepthToSpace, DepthToSpaceCpu);
} // namespace vknn
