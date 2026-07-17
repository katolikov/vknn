// The per-op capability table (see op_descriptor.h). One row per OpType is the single place an op's
// OpType-keyed facts live; gpuFlatNode (layout) and fusePointwiseChains (pwMember/pwEpilogue) read
// this table instead of each keeping a parallel OpType switch.
//
// LayoutClass::Flat  -> gpuFlatNode returns true for every node of this op.
// LayoutClass::ShapeDependent -> gpuFlatNode keeps a per-node predicate (Concat/Split/Softmax/Pad/
//   ConvTranspose/Gather/ScatterND/TopK/Einsum/Binary/Add/FusedPointwise); the descriptor records
//   only that a predicate exists.
// LayoutClass::Nc4 (default) -> gpuFlatNode returns false (NC4HW4: pointwise ops, CPU-only ops).
//
// pwMember   -> the per-element op types the fusion pass grows a unit from (its own float/shape/bound
//               checks still run per node).
// pwEpilogue -> op types whose kernel family has an _epi store variant that can host a fused unit.
#include "vknn/op_descriptor.h"

namespace vknn {

    namespace {
        using L = LayoutClass;

        // The largest enum value with a table entry (the enum is append-only, so this only grows).
        constexpr int kMaxOp = (int) OpType::ChannelShuffle;

        struct Table {
            OpDescriptor d[kMaxOp + 1];
            Table() {
                auto set = [&](OpType t, L layout, bool pwMember, bool pwEpilogue) {
                    OpDescriptor &e = d[(int) t];
                    e.layout        = layout;
                    e.pwMember      = pwMember;
                    e.pwEpilogue    = pwEpilogue;
                };
                // layout                          pwMember pwEpilogue
                set(OpType::Conv, L::Nc4, false, true);
                set(OpType::ConvTranspose, L::ShapeDependent, false, true);
                set(OpType::Clip, L::Flat, true, false);
                set(OpType::Relu, L::Nc4, true, false);
                set(OpType::Add, L::ShapeDependent, true, false);
                set(OpType::GlobalAvgPool, L::Nc4, false, true);
                set(OpType::AvgPool, L::Nc4, false, true);
                set(OpType::MaxPool, L::Nc4, false, true);
                set(OpType::Gemm, L::Nc4, false, true);
                set(OpType::MatMul, L::Flat, false, true);
                set(OpType::Einsum, L::ShapeDependent, false, false);
                set(OpType::Expand, L::Flat, false, false);
                set(OpType::Tile, L::Flat, false, false);
                set(OpType::Softmax, L::ShapeDependent, false, true);
                set(OpType::LayerNorm, L::Flat, false, true);
                set(OpType::RMSNorm, L::Flat, false, true);
                set(OpType::Concat, L::ShapeDependent, false, true);
                set(OpType::Pad, L::ShapeDependent, false, false);
                set(OpType::Gather, L::ShapeDependent, false, false);
                set(OpType::Unary, L::Nc4, true, false);
                set(OpType::Binary, L::ShapeDependent, true, false);
                set(OpType::PRelu, L::Nc4, true, false);
                set(OpType::Resize, L::Nc4, false, true);
                set(OpType::GridSample, L::Nc4, false, true);
                set(OpType::Transpose, L::Flat, false, true);
                set(OpType::Slice, L::Flat, false, true);
                set(OpType::Reduce, L::Flat, false, true);
                set(OpType::DepthToSpace, L::Flat, false, false);
                set(OpType::Split, L::ShapeDependent, false, false);
                set(OpType::Where, L::Flat, true, false);
                set(OpType::Equal, L::Flat, true, false);
                set(OpType::Greater, L::Flat, true, false);
                set(OpType::GreaterEqual, L::Flat, true, false);
                set(OpType::Less, L::Flat, true, false);
                set(OpType::LessEqual, L::Flat, true, false);
                set(OpType::ConstantOfShape, L::Flat, false, false);
                set(OpType::Range, L::Flat, false, false);
                set(OpType::ScatterND, L::ShapeDependent, false, false);
                set(OpType::FusedDwPw, L::Nc4, false, true);
                set(OpType::FusedPointwise, L::ShapeDependent, false, false);
                set(OpType::ConvGemm, L::Nc4, false, true);
                set(OpType::TopK, L::ShapeDependent, false, false);
                set(OpType::QuantizeLinear, L::Flat, false, false);
                set(OpType::DequantizeLinear, L::Flat, false, false);
                // Bool-producing flat elementwise ops (float->bool NaN test / broadcasting boolean AND).
                // Own flat kernels rather than fusion members: no pw step code is defined for them.
                set(OpType::IsNaN, L::Flat, false, false);
                set(OpType::And, L::Flat, false, false);
                // Fused rotate-half rotary embedding (fuseRope, load-time): a flat row-major
                // pointwise kernel with a cos/sin table row lookup; no fusion role of its own.
                set(OpType::Rope, L::Flat, false, false);
                // Load-time fused decode attention: operands and output are flat row-major
                // activations addressed through per-axis strides; never a pointwise-fusion member
                // or epilogue host (created after fusePointwiseChains has run).
                set(OpType::FusedAttention, L::Flat, false, false);
                // Group-interleave channel permutation (fuseChannelShuffle, import-time). Has a
                // kernel in BOTH layouts, so its layout is per-node: it adopts the input tensor's
                // assigned layout (the Agnostic arm in globalLayoutAssign) and never forces a
                // convert. Pure data movement; no fusion role.
                set(OpType::ChannelShuffle, L::ShapeDependent, false, false);
                // Everything not listed keeps the all-default row {Nc4, pwMember=false,
                // pwEpilogue=false}: CPU-only / structural ops (Reshape, Flatten, Squeeze, Unsqueeze,
                // Cast, Identity, Constant, Shape, BatchNorm, EyeLike, FusedSE, ConvertLayout,
                // ConvertDtype, Dropout, InstanceNorm, and the quantized QLinear/dynamic family) —
                // none runs on the flat path or takes part in pointwise fusion.
            }
        };
    } // namespace

    const OpDescriptor &opDescriptor(OpType t) {
        static const Table table;
        int                i = (int) t;
        if (i < 0 || i > kMaxOp)
        {
            static const OpDescriptor kDefault;
            return kDefault;
        }
        return table.d[i];
    }

} // namespace vknn
