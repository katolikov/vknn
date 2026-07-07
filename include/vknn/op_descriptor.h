// Per-op capability descriptor: the single source for the OpType-keyed facts that the layout
// classifier and the pointwise-fusion passes each used to keep as their own hand-synced switch.
#pragma once
#include "vknn/op_type.h"

namespace vknn {

    /// How an op's GPU kernel reads its tensors, for the NC4HW4-vs-flat layout assignment
    /// (gpuFlatNode, insert_layout_converts.cpp).
    enum class LayoutClass {
        /// NC4HW4 (channels packed in vec4 blocks) — the CNN default. Every pointwise op and every
        /// op with no flat kernel lands here; gpuFlatNode returns false.
        Nc4,
        /// Flat row-major at any rank, for every node of this op — gpuFlatNode returns true
        /// unconditionally (generic N-D gather/broadcast/reduce/matmul/compare kernels).
        Flat,
        /// The layout depends on the node's shapes/attributes (e.g. Concat that isn't 4D channel-axis
        /// 4-aligned goes flat, otherwise NC4HW4). gpuFlatNode keeps the per-node predicate for these;
        /// the descriptor only records that a predicate exists so the anti-drift test skips them.
        ShapeDependent,
    };

    /// The OpType-keyed capability facts an op declares once, consulted by the layout classifier and
    /// the pointwise-fusion pass so those consumers stop maintaining their own parallel OpType
    /// switches. Shape/attribute/dtype gating stays in code (vkNodeGate, the ShapeDependent
    /// gpuFlatNode arms, pwEligibleNode's per-node checks) — the descriptor carries only the facts
    /// that are a pure function of the OpType.
    struct OpDescriptor {
        /// GPU layout class (see LayoutClass). Default Nc4 covers pointwise ops and CPU-only ops.
        LayoutClass layout = LayoutClass::Nc4;
        /// True for the per-element op types eligible to join a fused-pointwise unit (the OpType gate
        /// of pwEligibleNode; the node still passes its own float-dtype/shape/bound checks).
        bool pwMember = false;
        /// True for op types whose GPU kernel family carries an _epi variant that can apply a
        /// pointwise unit at its store (pwEpilogueCapable). Anything else keeps a standalone unit.
        bool pwEpilogue = false;
    };

    /// The descriptor for `t`. Returns a reference into a static table; every OpType has an entry
    /// (an unrecognized value returns the all-default descriptor). Defined in op_descriptor.cpp,
    /// the one place an op's capability facts are declared.
    const OpDescriptor &opDescriptor(OpType t);

} // namespace vknn
