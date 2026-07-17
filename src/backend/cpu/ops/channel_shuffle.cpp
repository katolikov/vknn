// ChannelShuffle: group-interleave channel permutation (ShuffleNetV2), NCHW fp32 reference.
// Equivalent to Reshape([N,C,sp] -> [N,g,C/g,sp]) + Transpose(0,2,1,...) + Reshape back:
// the split writes channel c to group coords (i, j) = (c / (C/g), c % (C/g)), the transpose
// reorders to (j, i), and the merge flattens to c_out = j * g + i. Inverting for a gather,
// output channel c reads input channel (c % g) * (C/g) + c / g. Pure data movement over the
// spatial tail (any rank, including none), so the output is bit-identical to the folded chain.
#include "backend/cpu/cpu_backend.h"
#include "vknn/op.h"

namespace vknn {
    namespace {
        struct ChannelShuffleCpu: CpuOp {
            void run(const Node &node, ExecContext &ctx) override {
                const RtTensor &X          = ctx.t(node.inputs[0]);
                RtTensor       &Y          = ctx.t(node.outputs[0]);
                int64_t         groupCount = node.attr.geti("groups", 1);
                float          *out        = cpu::allocOut(Y, X.shape);
                const float    *in         = X.host.f32();
                int64_t         batch      = X.shape.size() > 0 ? X.shape[0] : 1;
                int64_t         channels   = X.shape.size() > 1 ? X.shape[1] : 1;
                // Spatial element count: product of every dim after the channel axis (1 for [N,C]).
                int64_t spatial = 1;
                for (size_t k = 2; k < X.shape.size(); ++k)
                {
                    spatial *= X.shape[k];
                }
                // A degenerate groups value (< 1, or one that does not divide C) would make the
                // permutation below read out of range; fall back to the identity copy, which is what
                // the unfolded Reshape chain would refuse to build in the first place.
                if (groupCount < 1 || channels % groupCount != 0)
                {
                    groupCount = 1;
                }
                int64_t groupWidth = channels / groupCount;
                for (int64_t n = 0; n < batch; ++n)
                {
                    for (int64_t c = 0; c < channels; ++c)
                    {
                        // Gather form of the derivation above: output channel c = j*g + i reads
                        // input channel i*(C/g) + j, with i = c % g and j = c / g.
                        int64_t      sourceChannel = (c % groupCount) * groupWidth + c / groupCount;
                        const float *src           = in + (n * channels + sourceChannel) * spatial;
                        float       *dst           = out + (n * channels + c) * spatial;
                        for (int64_t s = 0; s < spatial; ++s)
                        {
                            dst[s] = src[s];
                        }
                    }
                }
            }
        };
    } // namespace
    VKNN_REGISTER_CPU_OP(OpType::ChannelShuffle, ChannelShuffleCpu);
} // namespace vknn
