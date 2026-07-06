// Vulkan capability model (see vk_gates.h). vkNodeGate is the single shape-aware gate behind
// VulkanBackend::supportsNode: Concat and Binary only run on the GPU for the NC4HW4-friendly
// cases, constant-operand requirements keep runtime-parameter variants on the (always-correct)
// CPU op, and the flat row-major kernels bound the decodable rank. Each refusal names its gate so
// fallback diagnostics and the support report state WHY a node left the GPU.
#include "core/vk_gates.h"
#include "backend/cpu/cpu_backend.h"
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
            // CPU-only kernels.
            case OpType::TopK:
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
                return false;
            default:
                return true;
        }
    }

    bool vkNodeGate(const Graph &g, const Node &nd, std::string *whyNot) {
        // Generic N-D ops the GPU runs flat (Transpose/Slice always; Concat/Softmax/Binary/Add either
        // NC4HW4 or flat per the layout pass). The flat row-major kernels handle rank <= 6.
        if (nd.type == OpType::Transpose || nd.type == OpType::Slice || nd.type == OpType::ConvertLayout || nd.type == OpType::Concat || nd.type == OpType::Softmax || nd.type == OpType::Squeeze)
        {
            return true;
        }
        if (nd.type == OpType::Expand || nd.type == OpType::Tile)
        {
            // flat broadcast/tile gather decodes up to kMaxRank=6 output dims.
            if (g.desc(nd.outputs[0]).shape.size() <= 8)
            {
                return true;
            }
            return refuse(whyNot, std::string(opTypeName(nd.type)) + ": output rank > 8");
        }
        if (nd.type == OpType::Pad)
        {
            // Flat pad runs on the GPU only for static pads + a supported mode (else CPU). Mirrors
            // gpuFlatNode so a GPU-assigned Pad is always marked flat by the layout pass.
            if (g.desc(nd.outputs[0]).shape.size() > 8)
            {
                return refuse(whyNot, "Pad: output rank > 8");
            }
            std::string mode = nd.attr.gets("mode", "constant");
            if (mode != "constant" && mode != "edge" && mode != "reflect")
            {
                return refuse(whyNot, "Pad: mode '" + mode + "' unsupported (constant/edge/reflect only)");
            }
            bool padsKnown = !nd.attr.getints("pads").empty() || (nd.inputs.size() > 1 && nd.inputs[1] != kNoTensor && g.isInitializer(nd.inputs[1]));
            if (!padsKnown)
            {
                return refuse(whyNot, "Pad: runtime pads input");
            }
            if (pwCoreInputs(nd) > 2 && nd.inputs[2] != kNoTensor && !g.isInitializer(nd.inputs[2]))
            {
                return refuse(whyNot, "Pad: runtime pad-value input");
            }
            return true;
        }
        if (nd.type == OpType::MatMul)
        {
            // Batched N-D matmul on the flat row-major path; the kernel decodes up to kMaxRank=6 out
            // dims. Two operands, or three when a rank-1 bias is fused in (the _bias kernel binds it
            // as a 4th buffer). Inputs from pw_opbase on are fused pointwise-epilogue operands (bound
            // after the core buffers), not matmul operands.
            size_t core = nd.attr.has("pw_steps") ? (size_t) nd.attr.geti("pw_opbase", (int64_t) nd.inputs.size()) : nd.inputs.size();
            if (!(core == 2 || (core == 3 && nd.fusedBias != kNoTensor)))
            {
                return refuse(whyNot, "MatMul: operand count not 2 (or 3 with fused bias)");
            }
            if (g.desc(nd.outputs[0]).shape.size() <= 8)
            {
                return true;
            }
            return refuse(whyNot, "MatMul: output rank > 8");
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
            // flat reduce kernel: one thread per output element, loops the reduced axes. rank <= 6.
            const Shape &in = g.desc(nd.inputs[0]).shape;
            if (!in.empty() && in.size() <= 8)
            {
                return true;
            }
            return refuse(whyNot, "Reduce: input rank unresolved or > 8");
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
            // The plan-time output size is the fill count; integer-valued fills (int64 index
            // tensors) stay on the exact CPU op, as does an unresolved output size.
            if (g.desc(nd.outputs[0]).shape.empty() || g.desc(nd.outputs[0]).dtype != DType::Float32)
            {
                return refuse(whyNot, "ConstantOfShape: unresolved output shape or non-fp32 output");
            }
            auto it = nd.attr.map.find("value");
            if (it == nd.attr.map.end() || it->second.kind != Attr::Ints)
            {
                return true;
            }
            return refuse(whyNot, "ConstantOfShape: integer fill value");
        }
        if (nd.type == OpType::Range)
        {
            // The static plan fixes the output size; the scalar values may still be runtime
            // (read from their buffers at dispatch). int64 ranges (index vectors) const-fold or
            // stay on the exact CPU op, as does a Range whose size cannot resolve at plan time.
            if (nd.inputs.size() < 3 || g.desc(nd.outputs[0]).shape.empty())
            {
                return refuse(whyNot, "Range: fewer than 3 inputs or unresolved output shape");
            }
            if (g.desc(nd.outputs[0]).dtype != DType::Float32)
            {
                return refuse(whyNot, "Range: non-fp32 output");
            }
            for (int k = 0; k < 3; ++k)
            {
                // fp16 covers the scalars a --fp16 compile retyped; the upload decodes them.
                if (nd.inputs[k] == kNoTensor || (g.desc(nd.inputs[k]).dtype != DType::Float32 && g.desc(nd.inputs[k]).dtype != DType::Float16))
                {
                    return refuse(whyNot, "Range: non-float scalar input");
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
            const Shape &in   = g.desc(nd.inputs[0]).shape;
            const Shape &grid = g.desc(nd.inputs[1]).shape;
            if (in.size() != 4 || grid.size() != 4 || grid.back() != 2)
            {
                return refuse(whyNot, "GridSample: data/grid not 4D with trailing grid dim 2");
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
            // Flat reduction over the trailing axes; scale (and bias, if present) must be const
            // initializers.
            if (nd.inputs.size() < 2 || !g.isInitializer(nd.inputs[1]))
            {
                return refuse(whyNot, "LayerNorm: runtime scale input");
            }
            if (pwCoreInputs(nd) > 2 && nd.inputs[2] != kNoTensor && !g.isInitializer(nd.inputs[2]))
            {
                return refuse(whyNot, "LayerNorm: runtime bias input");
            }
            return true;
        }
        if (nd.type == OpType::Where || nd.type == OpType::Equal || nd.type == OpType::Greater || nd.type == OpType::GreaterEqual || nd.type == OpType::Less || nd.type == OpType::LessEqual)
        {
            // flat broadcasting kernels (fixed PC arrays) decode up to kMaxRank=8 output dims.
            if (g.desc(nd.outputs[0]).shape.size() <= 8)
            {
                return true;
            }
            return refuse(whyNot, std::string(opTypeName(nd.type)) + ": output rank > 8");
        }
        if (nd.type == OpType::ConvTranspose)
        {
            // Flat transposed conv: 4D input + constant weight (uploaded flat). A runtime weight or
            // non-4D input falls back to the CPU op.
            if (g.desc(nd.inputs[0]).shape.size() != 4)
            {
                return refuse(whyNot, "ConvTranspose: input not 4D");
            }
            if (nd.inputs.size() < 2 || !g.isInitializer(nd.inputs[1]))
            {
                return refuse(whyNot, "ConvTranspose: runtime weight input");
            }
            if (pwCoreInputs(nd) > 2 && nd.inputs[2] != kNoTensor && !g.isInitializer(nd.inputs[2]))
            {
                return refuse(whyNot, "ConvTranspose: runtime bias input");
            }
            return true;
        }
        if (nd.type == OpType::Unsqueeze)
        {
            return true; // metadata copy on the flat path
        }
        if (nd.type == OpType::Cast)
        {
            // float->float is a no-op copy; float->int truncates+clamps on the flat path (cast.comp),
            // carrying the logical integer as compute-precision storage. A cast whose INPUT is an int64
            // (CPU-only) shape/index tensor stays on the CPU op — the GPU has no int64 buffer to read.
            if (g.desc(nd.inputs[0]).dtype != DType::Int64)
            {
                return true;
            }
            return refuse(whyNot, "Cast: int64 input");
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
            // flat scatter; index may be a constant or a runtime float activation. Data rank within
            // kMaxRank.
            if (nd.inputs.size() >= 3 && g.desc(nd.inputs[0]).shape.size() <= 8)
            {
                return true;
            }
            return refuse(whyNot, "ScatterND: fewer than 3 inputs or data rank > 8");
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
            // per-channel affine; needs 4D input and the 4 params (gamma/beta/mean/var) as constants.
            if (nd.inputs.size() < 5 || g.desc(nd.inputs[0]).shape.size() != 4)
            {
                return refuse(whyNot, "BatchNorm: fewer than 5 inputs or input not 4D");
            }
            for (int i = 1; i <= 4; ++i)
            {
                if (!g.isInitializer(nd.inputs[i]))
                {
                    return refuse(whyNot, "BatchNorm: runtime scale/bias/mean/var input");
                }
            }
            return true;
        }
        if (nd.type == OpType::Split)
        {
            // NC4HW4 channel split (4D, axis 1, 4-aligned outputs) is a block copy; any other split runs
            // on the flat row-major path (a Slice per output) for rank <= kMaxRank.
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
            if (rank <= 8)
            {
                return true;
            }
            return refuse(whyNot, "Split: input rank > 8");
        }
        if (nd.type == OpType::Clip)
        {
            // const-or-absent scalar bounds (baked into the PC in prepare); runtime bounds fall back.
            for (int i = 1; i < 3 && i < (int) nd.inputs.size(); ++i)
            {
                if (nd.inputs[i] != kNoTensor && !g.isInitializer(nd.inputs[i]))
                {
                    return refuse(whyNot, "Clip: runtime min/max input");
                }
            }
            return true;
        }
        if (nd.type == OpType::QuantizeLinear || nd.type == OpType::DequantizeLinear)
        {
            // Flat affine dequant/quant: scale (input[1]) and any zero_point (input[2]) must be
            // constant initializers (uploaded flat in prepare); a runtime scale/zp falls back to the
            // exact CPU op. The flat kernel decodes the per-axis stride within the rank-8 bound.
            if (nd.inputs.size() < 2 || nd.inputs[1] == kNoTensor || !g.isInitializer(nd.inputs[1]))
            {
                return refuse(whyNot, std::string(opTypeName(nd.type)) + ": runtime scale input");
            }
            if (nd.inputs.size() > 2 && nd.inputs[2] != kNoTensor && !g.isInitializer(nd.inputs[2]))
            {
                return refuse(whyNot, std::string(opTypeName(nd.type)) + ": runtime zero_point input");
            }
            if (g.desc(nd.outputs[0]).shape.size() > 8)
            {
                return refuse(whyNot, std::string(opTypeName(nd.type)) + ": output rank > 8");
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
        // Conv: the GPU kernels cover group == 1 and pure depthwise (group == Cin == Cout).
        // Partial groups (1 < group < Cin) and depthwise with a channel multiplier
        // (group == Cin, Cout != Cin) would mis-index the dense [Cout][Cinb][KH][KW][4] weight
        // pack — those fall back to the group-aware CPU op.
        if (nd.type == OpType::Conv && nd.inputs.size() >= 2)
        {
            int64_t group = nd.attr.geti("group", 1);
            if (group > 1)
            {
                const Shape &xs = g.desc(nd.inputs[0]).shape;
                const Shape &ws = g.desc(nd.inputs[1]).shape;
                if (xs.size() == 4 && ws.size() == 4 && group == xs[1] && ws[0] == xs[1])
                {
                    return true;
                }
                return refuse(whyNot, "Conv: grouped conv is GPU-supported only as pure depthwise");
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
