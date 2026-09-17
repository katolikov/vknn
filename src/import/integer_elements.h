// Which tensors hold integer element values, and which of those the CPU backend stores as int64, resolved
// from the graph IR.
//
// The IR types graph inputs, graph outputs and initializers; a runtime intermediate carries a dtype label
// only where an import rule stamps one (ArgMax/ArgMin and TopK indices), so the label of an intermediate
// does not say whether it holds integers. These resolvers read that from the producers instead, and their
// readers share them so every rule that depends on an element type agrees on every tensor:
//  - ElementFact::IntegerValues (the ONNX element type is an integer) selects the Mod kernels' integer
//    mode (modOperandsAreInteger) and keeps a member out of a fused pointwise unit, which computes float
//    math on one storage kind and which the integer fp32 pin cannot pass through;
//  - ElementFact::Int64Storage mirrors the CPU kernels' runtime storage choice, which decides whether the
//    CPU op divides and raises in int64 (vkNodeGate keeps such a Div or Pow on the CPU op) and which the
//    fused CPU kernel cannot read (it reads every operand as fp32 lanes).
#pragma once
#include "vknn/graph.h"
#include <cstdint>
#include <vector>

namespace vknn {

    /// tensorProducers entry of a tensor no node writes (a graph input, an initializer, an orphan).
    inline constexpr int kNoTensorProducer = -1;

    /// Index of the node writing each tensor (the last writer when several do), kNoTensorProducer for a
    /// tensor no node writes. Sized to `g.tensors`.
    std::vector<int> tensorProducers(const Graph &g);

    /// The element fact an ElementFactResolver answers.
    enum class ElementFact : uint8_t {
        /// The tensor holds values of an ONNX integer element type. Holds for a tensor labeled Int64,
        /// Int32, Int8 or UInt8, and for the output of: a Cast to an integer type; Shape, ArgMax, ArgMin,
        /// BitShift, BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot; a Constant or ConstantOfShape with an
        /// integer value; TopK's indices. It holds for the output of an op whose result has its data
        /// operands' element type when one of those operands holds integers: layout and dtype converts,
        /// Identity, Reshape, Flatten, Squeeze, Unsqueeze, Slice, Transpose, Expand, Tile, Split, Gather,
        /// DepthToSpace, ChannelShuffle, Relu, Neg, Abs, Clip, ReduceSum/Max/Min/Prod and TopK's values
        /// (operand 0); Pad and ScatterND (operands 0 and 2); Range (operands 0 to 2); Where (operands 1
        /// and 2); Concat, Add and Mod (every operand); Binary (every operand, Pow its base alone).
        IntegerValues,
        /// The CPU backend stores the tensor as int64. Holds for a tensor labeled Int64 (a graph input or
        /// initializer binds so; an intermediate's label is taken as declared), and for the output of: a
        /// Cast to an integer type or BOOL; Shape, ArgMax, ArgMin; a Constant or ConstantOfShape with an
        /// integer value; TopK's indices. It holds for the output of: layout and dtype converts, Identity,
        /// Reshape, Flatten, Squeeze, Unsqueeze, Slice, Transpose, Expand, Tile, Split, Gather, ScatterND,
        /// BitwiseNot, ReduceSum/Max/Min/Prod and TopK's values when operand 0 is int64; Add, Mod,
        /// BitShift, BitwiseAnd, BitwiseOr, BitwiseXor and Binary (Pow: its base alone) when an operand is
        /// int64; Concat when a part is, Where when a value is; Range when all three operands are.
        Int64Storage,
    };

    /// Resolves one ElementFact tensor by tensor over a graph, walking producers and memoizing every
    /// tensor it visits. Every other tensor lacks the fact: a graph input or initializer whose label does
    /// not carry it, a Cast to a float type, a producer hosting a fused pointwise chain (pw_steps: its
    /// steps compute float values), and every op not listed for the fact. The graph must not change while
    /// the resolver is in use.
    class ElementFactResolver {
      public:
        /// Resolver over `g` with its own producer index.
        ElementFactResolver(const Graph &g, ElementFact fact);

        /// Resolver over `g` reading `producers`, tensorProducers(g) or an equivalent last-writer index.
        ElementFactResolver(const Graph &g, ElementFact fact, std::vector<int> producers);

        /// Whether `tensor` has the fact; false for kNoTensor and an out-of-range id.
        bool holds(TensorId tensor);

      private:
        enum class Resolution : uint8_t { Unresolved, Pending, Lacks, Holds };

        const Graph            &graph_;
        const ElementFact       fact_;
        std::vector<int>        producers_;
        std::vector<Resolution> resolution_;
    };

    /// Per-tensor answer of `fact` for every tensor of `g` (1 when the tensor has it), sized to `g.tensors`.
    std::vector<char> resolveElementFact(const Graph &g, ElementFact fact);

} // namespace vknn
