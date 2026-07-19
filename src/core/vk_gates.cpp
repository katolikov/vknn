// Vulkan capability model (see vk_gates.h). vkNodeGate is the single shape-aware gate behind
// VulkanBackend::supportsNode: Concat and Binary only run on the GPU for the NC4HW4-friendly
// cases, constant-operand requirements keep runtime-parameter variants on the (always-correct)
// CPU op, and the flat row-major kernels bound the decodable rank. Each refusal names its gate so
// fallback diagnostics and the support report state WHY a node left the GPU.
#include "core/vk_gates.h"
#include "backend/cpu/cpu_backend.h"
#include "core/fused_attention.h"
#include "vknn/dtype.h"
#include "vknn/node.h"

namespace vknn {

    namespace {
        // Fills the refusal reason when the caller asked for one; the return value stays the only
        // control signal, so a null whyNot costs nothing.
        inline bool refuse(std::string *whyNot, const std::string &reason) {
            if (whyNot)
            {
                *whyNot = reason;
            }
            return false;
        }
    } // namespace

    bool vkKernelDeclared(OpType t) {
        switch (t)
        {
            // Const-fold / import-time ops with the CPU op as the terminal fallback.
            case OpType::Unknown:
            case OpType::Identity:
            case OpType::Constant:
            case OpType::Shape:
            case OpType::EyeLike:
            // Erased/lowered at import; a survivor has no kernel in either backend.
            case OpType::Dropout:
            case OpType::InstanceNorm:
            // ONNX quantized family that still has no direct kernel: recognized at import for precise
            // reporting; execution goes through the import-time dequantize lowering. (Quantize/
            // DequantizeLinear DO have flat kernels — the graph-boundary dequant a genuine int input
            // needs — so they fall to the default `return true` and vkNodeGate below.)
            case OpType::DynamicQuantizeLinear:
            case OpType::QLinearConv:
            case OpType::QLinearMatMul:
            case OpType::QLinearAdd:
            case OpType::QLinearGlobalAveragePool:
            case OpType::MatMulInteger:
            case OpType::ConvInteger:
            case OpType::QGemm:
            // ORT contrib family: recognized at import for precise reporting; execution goes through
            // the lowerOrtContribOps expansion. A survivor (a variant the expansion declines) has
            // no kernel in either backend.
            case OpType::SimplifiedLayerNorm:
            case OpType::SkipSimplifiedLayerNorm:
            case OpType::SkipLayerNorm:
            case OpType::RotaryEmbedding:
            case OpType::MultiHeadAttention:
            case OpType::GroupQueryAttention:
            case OpType::MatMulNBits:
                return false;
            default:
                return true;
        }
    }

    bool vkNodeGate(const Graph &g, const Node &nd, std::string *whyNot) {
        // Generic N-D ops the GPU runs flat (Transpose/Slice always; Concat/Softmax/Binary/Add either
        // NC4HW4 or flat per the layout pass). The flat row-major kernels decode any rank: the per-axis
        // geometry rides a plan SSBO (flat::uploadFlatGeom), not the push constant, so there is no
        // rank ceiling on the decode.
        if (nd.type == OpType::Transpose || nd.type == OpType::Slice || nd.type == OpType::ConvertLayout || nd.type == OpType::Concat || nd.type == OpType::Softmax || nd.type == OpType::Squeeze)
        {
            return true;
        }
        if (nd.type == OpType::Expand || nd.type == OpType::Tile)
        {
            // flat broadcast/tile gather decodes any output rank (geometry in a plan SSBO).
            return true;
        }
        if (nd.type == OpType::Pad)
        {
            // Flat pad runs on the GPU only for static pads + a supported mode (else CPU). Mirrors
            // gpuFlatNode so a GPU-assigned Pad is always marked flat by the layout pass. The flat
            // kernel decodes any rank (geometry in a plan SSBO).
            std::string mode = nd.attr.gets("mode", "constant");
            if (mode != "constant" && mode != "edge" && mode != "reflect")
            {
                return refuse(whyNot, "Pad: mode '" + mode + "' unsupported (constant/edge/reflect only)");
            }
            bool padsKnown = !nd.attr.getints("pads").empty() || (nd.inputs.size() > 1 && nd.inputs[1] != kNoTensor && g.isInitializer(nd.inputs[1]));
            if (!padsKnown)
            {
                // Runtime pads make the output shape data-dependent (no static buffer plan), so the
                // pad GEOMETRY stays on the CPU op. (A runtime pad VALUE with static geometry runs on
                // the GPU: flat_pad_rt reads the fill value from an SSBO.)
                return refuse(whyNot, "Pad: runtime pads input");
            }
            return true;
        }
        if (nd.type == OpType::MatMul)
        {
            // Batched N-D matmul on the flat row-major path; the kernel decodes any output rank
            // (geometry in a plan SSBO). Two operands, or three when a rank-1 bias is fused in (the
            // _bias kernel binds it as a 4th buffer). Inputs from pw_opbase on are fused
            // pointwise-epilogue operands (bound after the core buffers), not matmul operands.
            size_t core = pwCoreInputs(nd); // hardened: clamps pw_opbase to [0, inputs.size()]
            if (!(core == 2 || (core == 3 && nd.fusedBias != kNoTensor)))
            {
                return refuse(whyNot, "MatMul: operand count not 2 (or 3 with fused bias)");
            }
            return true;
        }
        if (nd.type == OpType::Gemm)
        {
            // The GPU Gemm kernel computes only Y = op_transB(B)-form A*B with unit alpha/beta and no
            // transA. A Gemm carrying alpha/beta != 1 or transA=1 has no GPU path, so refuse it to the
            // CPU Gemm op (which honors all of alpha/beta/transA/transB) instead of silently dropping
            // those attributes and returning a wrong result. The common nn.Linear Gemm (alpha=beta=1,
            // transA=0, transB=1) is unaffected and stays on the GPU.
            if (nd.attr.getf("alpha", 1.f) != 1.f || nd.attr.getf("beta", 1.f) != 1.f || nd.attr.geti("transA", 0) != 0)
            {
                return refuse(whyNot, "Gemm: alpha/beta != 1 or transA set (GPU kernel handles only unit alpha/beta with transA=0)");
            }
            return true;
        }
        if (nd.type == OpType::DepthToSpace)
        {
            // [N,C,H,W] -> [N,C/b^2,H*b,W*b]; flat index-remap kernel. Need 4D and C divisible by b^2.
            const Shape &in = g.desc(nd.inputs[0]).shape;
            int64_t      b  = nd.attr.geti("blocksize", 1);
            if (in.size() == 4 && b >= 1 && in[1] % (b * b) == 0)
            {
                return true;
            }
            return refuse(whyNot, "DepthToSpace: input not 4D with C divisible by blocksize^2");
        }
        if (nd.type == OpType::Reduce)
        {
            // flat reduce kernel: one thread per output element, loops the reduced axes. Any input rank
            // (geometry in a plan SSBO); only an unresolved input shape stays on the CPU op.
            const Shape &in = g.desc(nd.inputs[0]).shape;
            if (!in.empty())
            {
                return true;
            }
            return refuse(whyNot, "Reduce: input rank unresolved");
        }
        if (nd.type == OpType::FusedDwPw)
        {
            // LDS holds E depthwise outputs (cap 1024). Run ALL eligible fused nodes on the GPU: a
            // partial gate (some fused nodes on CPU) creates a GPU/CPU boundary that mis-handles the
            // fused residual.
            const Shape &in  = g.desc(nd.inputs[0]).shape;  // expanded [N,E,H,W]
            const Shape &out = g.desc(nd.outputs[0]).shape; // [N,Cout,OH,OW]
            if (in.size() != 4 || out.size() != 4)
            {
                return refuse(whyNot, "FusedDwPw: input/output not 4D");
            }
            if (in[1] <= 1024)
            {
                return true;
            }
            return refuse(whyNot, "FusedDwPw: expanded channels > 1024");
        }
        if (nd.type == OpType::FusedSE)
        {
            // fixed LDS arrays: avg[1024], s1[256]
            const Shape &f  = g.desc(nd.inputs[0]).shape;
            const Shape &w1 = g.desc(nd.inputs[1]).shape;
            if (f.size() == 4 && f[1] <= 1024 && !w1.empty() && w1[0] <= 256)
            {
                return true;
            }
            return refuse(whyNot, "FusedSE: shape exceeds LDS caps (C <= 1024, squeeze <= 256)");
        }
        if (nd.type == OpType::ConstantOfShape)
        {
            // The plan-time output size is the fill count. An integer `value` fills the buffer with the
            // constant carried as compute-precision float — exact for the index/shape magnitudes these
            // produce — and the graph boundary repacks the declared int32/int64 dtype on readback. Only
            // an unresolved output size (data-dependent shape input) stays on the exact CPU op.
            if (g.desc(nd.outputs[0]).shape.empty())
            {
                return refuse(whyNot, "ConstantOfShape: unresolved output shape");
            }
            return true;
        }
        if (nd.type == OpType::Range)
        {
            // The static plan fixes the output size; the scalar values may still be runtime (read from
            // their buffers at dispatch). Integer start/limit/delta generate the ramp in compute-precision
            // float — exact for the index magnitudes these produce — and the graph boundary repacks the
            // declared int32/int64 dtype on readback. The int64/int32 scalars decode to float at the pack
            // boundary, so the shader reads them like any float operand. Only a Range whose output size
            // cannot resolve at plan time stays on the exact CPU op.
            if (nd.inputs.size() < 3 || g.desc(nd.outputs[0]).shape.empty())
            {
                return refuse(whyNot, "Range: fewer than 3 inputs or unresolved output shape");
            }
            for (int k = 0; k < 3; ++k)
            {
                if (nd.inputs[k] == kNoTensor)
                {
                    return refuse(whyNot, "Range: missing scalar input");
                }
            }
            return true;
        }
        if (nd.type == OpType::GridSample)
        {
            // 4D NC4HW4 data + a flat [N,Hout,Wout,2] grid (constant OR runtime — the layout pass keeps
            // the grid flat and the op binds it at compute precision).
            if (nd.inputs.size() < 2 || nd.inputs[1] == kNoTensor)
            {
                return refuse(whyNot, "GridSample: missing grid input");
            }
            const Shape &in = g.desc(nd.inputs[0]).shape;
            // Warp mode (fuseGridSampleWarp): 4D NC4HW4 data + an NCHW flow [N,2,Hout,Wout] (input 1)
            // + a base-grid constant [.,Hout,Wout,2] (input 2); the op computes coord = base +
            // scale*flow inside the sampler, so no [.,.,.,2] grid is materialized.
            if (nd.attr.has("warp"))
            {
                if (nd.inputs.size() < 3 || nd.inputs[2] == kNoTensor)
                {
                    return refuse(whyNot, "GridSample (warp): missing base-grid input");
                }
                const Shape &flow = g.desc(nd.inputs[1]).shape;
                const Shape &base = g.desc(nd.inputs[2]).shape;
                if (in.size() != 4 || flow.size() != 4 || flow[1] != 2 || base.size() != 4 || base.back() != 2)
                {
                    return refuse(whyNot, "GridSample (warp): data/flow/base not 4D with flow C==2 and base trailing dim 2");
                }
            } else
            {
                const Shape &grid = g.desc(nd.inputs[1]).shape;
                if (in.size() != 4 || grid.size() != 4 || grid.back() != 2)
                {
                    return refuse(whyNot, "GridSample: data/grid not 4D with trailing grid dim 2");
                }
            }
            std::string m = nd.attr.gets("mode", "bilinear");
            if (m == "bilinear" || m == "linear" || m == "nearest" || m == "cubic" || m == "bicubic")
            {
                return true;
            }
            return refuse(whyNot, "GridSample: mode '" + m + "' unsupported (bilinear/linear/nearest/cubic/bicubic only)");
        }
        if (nd.type == OpType::Resize)
        {
            // GPU kernel resizes spatial dims only; channel/batch resize falls back to the CPU op.
            const Shape &in  = g.desc(nd.inputs[0]).shape;
            const Shape &out = g.desc(nd.outputs[0]).shape;
            if (in.size() == 4 && out.size() == 4 && in[0] == out[0] && in[1] == out[1])
            {
                return true;
            }
            return refuse(whyNot, "Resize: only 4D spatial resize (N and C unchanged)");
        }
        if (nd.type == OpType::LayerNorm)
        {
            // Flat reduction over the trailing axes. A constant scale/bias uploads flat; a runtime
            // scale/bias binds its activation buffer at dispatch (the shader reads gamma[j]/beta[j] the
            // same way). Only the presence of the scale input is required.
            if (nd.inputs.size() < 2 || nd.inputs[1] == kNoTensor)
            {
                return refuse(whyNot, "LayerNorm: missing scale input");
            }
            return true;
        }
        if (nd.type == OpType::RMSNorm)
        {
            // Flat sum-of-squares reduction over the trailing axis. The gamma (scale) input is a 1-D
            // [norm] tensor uploaded flat in prepare(); its presence is the only requirement.
            if (nd.inputs.size() < 2 || nd.inputs[1] == kNoTensor)
            {
                return refuse(whyNot, "RMSNorm: missing scale input");
            }
            return true;
        }
        if (nd.type == OpType::Rope)
        {
            // Fused rotate-half rotary embedding: pointwise over the flat input plus a cos/sin
            // table row lookup by position. Created only by fuseRope, which already validated the
            // shapes; the four inputs (x, positions, cos table, sin table) are the only requirement.
            if (nd.inputs.size() < 4 || nd.inputs[1] == kNoTensor || nd.inputs[2] == kNoTensor || nd.inputs[3] == kNoTensor)
            {
                return refuse(whyNot, "Rope: missing position/table input");
            }
            return true;
        }
        if (nd.type == OpType::Where || nd.type == OpType::Equal || nd.type == OpType::Greater || nd.type == OpType::GreaterEqual || nd.type == OpType::Less || nd.type == OpType::LessEqual || nd.type == OpType::And)
        {
            // flat broadcasting kernels decode any output rank (geometry in a plan SSBO).
            return true;
        }
        if (nd.type == OpType::FusedAttention)
        {
            // Load-time fused decode attention (core/fused_attention.h). The fuseDecodeAttention
            // pass only emits fully-formed nodes, so the structural checks here guard against a
            // hand-built graph; the device-dependent subgroup checks live in
            // VulkanBackend::supportsNode (this gate is device-free).
            if (!nd.attr.has(kFa))
            {
                return refuse(whyNot, "FusedAttention: missing geometry attrs");
            }
            const int64_t hd = nd.attr.geti(kFaHd), c = nd.attr.geti(kFaC);
            if (hd <= 0 || hd > kFaMaxHeadDim || c <= 0 || c > kFaMaxContext)
            {
                return refuse(whyNot, "FusedAttention: head dim / token count out of kernel range");
            }
            if (nd.inputs.size() < 3)
            {
                return refuse(whyNot, "FusedAttention: needs q, k, v inputs");
            }
            return true;
        }
        if (nd.type == OpType::ConvTranspose)
        {
            // Flat transposed conv: 4D input. A constant weight/bias uploads flat; a runtime weight/bias
            // binds its activation buffer at dispatch (the shader indexes ws[]/bs[] identically). A
            // non-4D input or a missing weight falls back to the CPU op.
            if (g.desc(nd.inputs[0]).shape.size() != 4)
            {
                return refuse(whyNot, "ConvTranspose: input not 4D");
            }
            if (nd.inputs.size() < 2 || nd.inputs[1] == kNoTensor)
            {
                return refuse(whyNot, "ConvTranspose: missing weight input");
            }
            return true;
        }
        if (nd.type == OpType::Unsqueeze)
        {
            return true; // metadata copy on the flat path
        }
        if (nd.type == OpType::Det)
        {
            // Batched determinant: the GPU kernel is a fixed-order cofactor expansion whose
            // register array covers n <= kDetMaxAnalyticN. The shape must be resolved and square;
            // larger matrices take the CPU's general LU (a loud, named fallback).
            const Shape &in = g.desc(nd.inputs[0]).shape;
            if (in.size() < 2)
            {
                return refuse(whyNot, "Det: input shape unresolved or rank < 2");
            }
            const int64_t rows = in[in.size() - 2], cols = in[in.size() - 1];
            if (rows != cols || cols < 1)
            {
                return refuse(whyNot, "Det: matrix must be square");
            }
            if (cols > kDetMaxAnalyticN)
            {
                return refuse(whyNot, "Det: n > kDetMaxAnalyticN runs on the CPU (partial-pivot LU)");
            }
            return true;
        }
        if (nd.type == OpType::ChannelShuffle)
        {
            // Group-interleave channel permutation; kernels exist in both layouts (flat / NC4HW4).
            // fuseChannelShuffle only emits valid nodes, so this gate guards a hand-built graph:
            // the shape must be resolved (the static plan sizes the dispatch from it) and the
            // channel count must split evenly into `groups`.
            const Shape &in = g.desc(nd.inputs[0]).shape;
            if (in.size() < 2)
            {
                return refuse(whyNot, "ChannelShuffle: input shape unresolved or rank < 2");
            }
            int64_t groupCount = nd.attr.geti("groups", 1);
            if (groupCount < 1 || in[1] % groupCount != 0)
            {
                return refuse(whyNot, "ChannelShuffle: groups must be >= 1 and divide the channel count");
            }
            return true;
        }
        if (nd.type == OpType::Cast)
        {
            // float->float is a no-op copy; float->int truncates+narrows on the flat path (cast.comp),
            // carrying the logical integer as compute-precision storage. An int64 INPUT (a shape/index
            // tensor) is decoded to compute-precision float when it is packed to the device (packToBuffer),
            // so cast.comp reads it like any other operand. A non-int64 input always runs on the GPU.
            if (g.desc(nd.inputs[0]).dtype != DType::Int64)
            {
                return true;
            }
            int64_t to = nd.attr.geti("to", 1); // ONNX TensorProto dtype
            // int64 -> FLOAT/FLOAT16/DOUBLE, INT32, INT64: the shape-arithmetic targets, exact in the
            // compute float. int64 -> INT8 (3) / UINT8 (2): cast.comp narrows to match the CPU Cast op
            // followed by the readback narrowing bit-for-bit (INT8 modulo-wrap, UINT8 saturate); the
            // narrowed value is small and exact in fp16/fp32. int64 -> BOOL (9): cast.comp truncates and
            // clamps to [0,1], bit-identical to the CPU op for the {0,1} mask tensors this targets.
            if (to == 1 || to == 10 || to == 11 || to == 6 || to == 7 || to == 2 || to == 3 || to == 9)
            {
                return true;
            }
            // int64 -> INT16/UINT16 (fp32 output tensor, no readback narrowing to lean on) / the 32/64-bit
            // unsigned targets keep the exact CPU op, where the wider range and modulo are exact.
            return refuse(whyNot, "Cast: int64 input to a narrow integer target");
        }
        if (nd.type == OpType::TopK)
        {
            // Per-slice selection along `axis` (flat row-major). Needs a resolved input shape and a
            // compile-time k: the `k` attribute (opset < 10) or a constant int64 input[1] (opset 10+),
            // baked into the push constant in prepare. A runtime k (non-initializer input[1]) stays on
            // the exact CPU op, since the static plan fixes the output slot count.
            if (g.desc(nd.inputs[0]).shape.empty())
            {
                return refuse(whyNot, "TopK: unresolved input shape");
            }
            if (nd.attr.has("k"))
            {
                return true;
            }
            if (pwCoreInputs(nd) > 1 && nd.inputs[1] != kNoTensor && g.isInitializer(nd.inputs[1]))
            {
                return true;
            }
            return refuse(whyNot, "TopK: runtime k input");
        }
        if (nd.type == OpType::Gather)
        {
            // flat axis-aware gather; index may be a constant (uploaded) or a runtime float activation
            // (RoPE).
            if (nd.inputs.size() >= 2)
            {
                return true;
            }
            return refuse(whyNot, "Gather: missing index input");
        }
        if (nd.type == OpType::ScatterND)
        {
            // flat scatter; index may be a constant or a runtime float activation. Any data rank
            // (geometry in a plan SSBO).
            if (nd.inputs.size() >= 3)
            {
                return true;
            }
            return refuse(whyNot, "ScatterND: fewer than 3 inputs");
        }
        if (nd.type == OpType::Einsum)
        {
            // Only "i,j->ij" (outer product) has a GPU kernel; other equations use the CPU op.
            std::string eq;
            for (char c: nd.attr.gets("equation", ""))
            {
                if (c != ' ' && c != '\t')
                {
                    eq += c;
                }
            }
            if (eq == "i,j->ij")
            {
                return true;
            }
            return refuse(whyNot, "Einsum: equation '" + eq + "' (GPU kernel covers i,j->ij only)");
        }
        if (nd.type == OpType::BatchNorm)
        {
            // per-channel affine over 4D input. Constant params (gamma/beta/mean/var) fold on the host;
            // runtime params bind as SSBOs and fold per channel in batchnorm_rt.comp. Needs the 5 inputs
            // present and a 4D data tensor.
            if (nd.inputs.size() < 5 || g.desc(nd.inputs[0]).shape.size() != 4)
            {
                return refuse(whyNot, "BatchNorm: fewer than 5 inputs or input not 4D");
            }
            for (int i = 1; i <= 4; ++i)
            {
                if (nd.inputs[i] == kNoTensor)
                {
                    return refuse(whyNot, "BatchNorm: missing scale/bias/mean/var input");
                }
            }
            return true;
        }
        if (nd.type == OpType::Split)
        {
            // NC4HW4 channel split (4D, axis 1, 4-aligned outputs) is a block copy; any other split runs
            // on the flat row-major path (a Slice per output) at any rank (geometry in a plan SSBO).
            const Shape &in = g.desc(nd.inputs[0]).shape;
            if (in.empty())
            {
                return refuse(whyNot, "Split: unresolved input shape");
            }
            int     rank = (int) in.size();
            int64_t axis = nd.attr.geti("axis", 0);
            if (axis < 0)
            {
                axis += rank;
            }
            if (rank == 4 && axis == 1)
            {
                bool aligned = true;
                for (TensorId o: nd.outputs)
                {
                    if (o == kNoTensor)
                    {
                        continue;
                    }
                    const Shape &os = g.desc(o).shape;
                    if (os.size() != 4 || os[1] % 4 != 0)
                    {
                        aligned = false;
                    }
                }
                if (aligned)
                {
                    return true;
                }
            }
            return true; // flat row-major split, any rank
        }
        if (nd.type == OpType::Clip)
        {
            // Constant/absent scalar bounds bake into the push constant (clip.comp); a runtime min/max
            // binds as an SSBO scalar read at dispatch (clip_rt.comp). Both run on the flat path.
            return true;
        }
        if (nd.type == OpType::QuantizeLinear || nd.type == OpType::DequantizeLinear)
        {
            // Flat affine dequant/quant: scale (input[1]) and any zero_point (input[2]) must be
            // constant initializers (uploaded flat in prepare); a runtime scale/zp falls back to the
            // exact CPU op. The kernel decodes the per-axis channel via a single `inner` stride scalar
            // (no per-rank push-constant arrays), so any rank runs on the GPU.
            if (nd.inputs.size() < 2 || nd.inputs[1] == kNoTensor || !g.isInitializer(nd.inputs[1]))
            {
                return refuse(whyNot, std::string(opTypeName(nd.type)) + ": runtime scale input");
            }
            if (nd.inputs.size() > 2 && nd.inputs[2] != kNoTensor && !g.isInitializer(nd.inputs[2]))
            {
                return refuse(whyNot, std::string(opTypeName(nd.type)) + ": runtime zero_point input");
            }
            return true;
        }
        // Add/Binary: 2 inputs required. The NC4HW4 kernel does same-shape + channel-broadcast; the
        // flat kernel (chosen by the layout pass) does everything else incl. constant operands.
        if (nd.type == OpType::Add || nd.type == OpType::Binary)
        {
            if (nd.inputs.size() == 2)
            {
                return true;
            }
            return refuse(whyNot, std::string(opTypeName(nd.type)) + ": input count != 2");
        }
        // Conv: the GPU kernels cover group == 1 (dense) and pure depthwise (group == Cin == Cout).
        // A general grouped conv (1 < group < Cin, e.g. ResNeXt cardinality, and the channel-multiplier
        // depthwise group == Cin/Cout != Cin) runs on the GPU too, but by import-time lowering into
        // `group` group-1 Convs over per-group channel slices joined by a Concat (lowerGroupedConv) —
        // not a dedicated grouped kernel. So a grouped Conv only reaches this gate when lowering could
        // not fire (a runtime/non-constant weight, or unresolved shapes), and it falls back to the
        // group-aware CPU op.
        if (nd.type == OpType::Conv && nd.inputs.size() >= 2)
        {
            int64_t group = nd.attr.geti("group", 1);
            if (group > 1)
            {
                const Shape &xs = g.desc(nd.inputs[0]).shape;
                const Shape &ws = g.desc(nd.inputs[1]).shape;
                if (xs.size() == 4 && ws.size() == 4 && group == xs[1] && ws[0] == xs[1])
                {
                    return true; // pure depthwise (group == Cin == Cout): native dwconv kernel
                }
                return refuse(whyNot, "Conv: unlowered grouped conv (runtime weight or unresolved shapes)");
            }
            return true;
        }
        return true;
    }

    std::vector<NodeSupport> vkSupportSurvey(const Graph &g) {
        std::vector<NodeSupport> rows;
        rows.reserve(g.nodes.size());
        for (const Node &nd: g.nodes)
        {
            NodeSupport row;
            row.node = nd.name;
            row.op   = opTypeName(nd.type);
            if (!vkKernelDeclared(nd.type))
            {
                bool cpuHas = CpuOpRegistry::instance().has(nd.type);
                row.backend = cpuHas ? "cpu" : "none";
                row.reason  = cpuHas ? "no vulkan kernel registered" : "no kernel in any backend";
            } else if (std::string why; !vkNodeGate(g, nd, &why))
            {
                row.backend = "cpu";
                row.reason  = why;
            } else
            {
                row.backend = "vulkan";
            }
            rows.push_back(std::move(row));
        }
        return rows;
    }

} // namespace vknn
