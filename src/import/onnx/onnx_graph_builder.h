// Builds a vknn Graph from a GraphProto in phases: collect() reads the proto (deferring node I/O to raw
// name lists), materializeInitializers() fills weight buffers, ssaResolveNodeIO() gives every node output
// a fresh TensorId (so a trace that reuses tensor names does not alias distinct tensors),
// stampIntegerWidths() records the ONNX integer width on the ops whose result depends on it, and
// dropInitializerInputs() cleans the input list.
#pragma once
#include "onnx_node_parser.h"
#include "onnx_reader.h"
#include "onnx_tensor_parser.h"
#include "onnx_types.h"
#include "vknn/graph.h"
#include "vknn/logging.h"
#include "vknn/op.h"
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace vknn { namespace onnx {

    class GraphBuilder {
        Graph             &g;
        const std::string &baseDir;
        struct PendingInit {
            std::string name;
            TensorProto tp;
        };
        std::vector<PendingInit>                    inits;
        std::map<std::string, std::vector<uint8_t>> extCache; // external .data files, read once each
        std::vector<Node>                           nodes;
        std::vector<NodeWireInfo>                   nodeWire;           // parallel to nodes: op_type spelling, tensor-attribute data_type
        std::vector<std::vector<std::string>>       nodeIns, nodeOuts;  // raw names, resolved in ssaResolveNodeIO
        std::map<std::string, Shape>                valueInfoShapes;    // value_info hints -> node outputs
        std::map<std::string, int32_t>              valueInfoElemTypes; // value_info element types -> node outputs
        // ONNX element type (TensorProto.DataType) declared on a tensor: a graph input or output, an
        // initializer, or a node output carrying an attributable value_info. The IR dtype keeps no
        // record of widths it does not store (UINT16, UINT32, ...), so the integer-width stamping reads
        // this instead.
        std::unordered_map<TensorId, int32_t> declaredElemTypes;

      public:
        GraphBuilder(Graph &graph, const std::string &dir): g(graph), baseDir(dir) {
        }

        void build(Reader r);

      private:
        // A declared shape is only trusted when fully static. dim_param (symbolic) dims parse
        // to -1; storing them as a desc would read as "resolved" and poison downstream
        // inference (a Reshape 0-copy of a -1, a Slice clamp against -1 -> 0, then a Shape()
        // fold freezes the lie). Graph inputs keep -1 dims: runStandardPasses substitutes the
        // static batch there.
        static bool fullyStatic(const Shape &sh);

        // Reads the GraphProto, dispatching on its protobuf field number: 1 = node (deferred to raw
        // name lists here; I/O is bound in ssaResolveNodeIO), 5 = initializer (deferred to inits;
        // weights are filled in materializeInitializers), 11 = graph input, 12 = graph output,
        // 13 = value_info (a shape hint, applied to its node output in ssaResolveNodeIO). Any other
        // field is skipped. Inputs/outputs are registered by name (findOrAdd); a symbolic output
        // shape is dropped (fullyStatic), an input keeps its -1 dims for the later batch substitution.
        void collect(Reader r);

        // Turns each initializer collected by collect() into a graph tensor with a host weight buffer.
        // Runs before ssaResolveNodeIO so initializer names resolve to real ids when node inputs bind.
        // External initializers (data held in a sibling file) are pulled in first via resolveExternal.
        // Storage dtype is narrowed to host representations: INT64 / UINT32 / UINT64 materialize to Int64
        // (a UINT64 value at or above 2^63 keeping its bit pattern); INT8 / UINT8 keep native 1-byte lanes;
        // every other type (FLOAT / FLOAT16 / DOUBLE / INT32 / INT16 / UINT16 / BOOL) materializes to fp32.
        // A scalar initializer has empty dims, so its element count is forced to 1.
        void materializeInitializers();

        // ONNX requires unique tensor names, but un-deduped PyTorch traces reuse them (e.g. two Cast
        // nodes both output "Cast_output_0"). Binding by name (findOrAdd) would alias distinct tensors
        // onto ONE TensorId with ONE shape -> wrong static buffer sizes / shape mismatches on the GPU
        // path. Instead: bind each node input to the nearest PRECEDING producer of that name, and give
        // each node output a FRESH TensorId (carrying its value_info shape hint). Declared graph outputs
        // re-point to their final producer.
        //
        // value_info hints are keyed by NAME, so a hint is attributable only when the name has exactly
        // ONE producer. On a reused name the hint belongs to (at most) one of the instances; stamping
        // it on all of them fabricates descs -- shape inference then builds on the lie (mis-sized
        // buffers, backend-support gates flipping to CPU, wrong output shapes).
        void ssaResolveNodeIO();

        // Stamps `int_bits` / `int_signed` (core/bitwise_attrs.h) on every BitShift and BitwiseNot node
        // from the ONNX element type of its input 0 (BitShift: input 1 when input 0 does not resolve).
        // A tensor's type is its declared type when it has one (declaredElemTypes); otherwise its producer
        // decides (producerElemType). A node whose operand type does not resolve to an integer type keeps
        // the attribute defaults (64-bit signed) and is named in one warning.
        void stampIntegerWidths();

        // Producer hops one element-type resolution follows before it gives up. Resolved types are
        // memoized along every walk, so the cap only bounds a malformed (cyclic) producer chain.
        static constexpr int kElemTypeResolveMaxHops = 4096;
        // Node index and output slot writing a tensor.
        struct ProducerSlot {
            size_t node;
            size_t output;
        };
        // Element type a producer fixes for the output `producer` names: Cast -> `to`; Shape / Size /
        // ArgMax / ArgMin / NonZero / NonMaxSuppression and TopK output 1 -> INT64; MatMulInteger /
        // ConvInteger -> INT32; the comparison and boolean ops (Equal ... IsInf) -> BOOL; Constant -> its
        // value tensor's type (INT64 for value_int / value_ints); ConstantOfShape -> its fill tensor's type
        // (FLOAT when absent); EyeLike with `dtype` -> that type; QuantizeLinear without a zero point ->
        // UINT8. Any other producer returns OnnxType::Undefined with `typeSource` set to the input whose
        // type the result takes (QuantizeLinear's zero point; Where / CastLike -> input 1; DequantizeLinear
        // -> its scale; OneHot -> its values; every other op -> input 0), or kNoTensor when absent.
        int32_t producerElemType(const ProducerSlot &producer, TensorId &typeSource) const;
        // ONNX element type of tensor `t` (OnnxType::Undefined when unresolved), memoized in `resolved`.
        int32_t resolveElemType(TensorId t, const std::unordered_map<TensorId, ProducerSlot> &producers, std::unordered_map<TensorId, int32_t> &resolved) const;

        // ONNX lists initializers in the graph input list too; drop them.
        void dropInitializerInputs();
    };

}} // namespace vknn::onnx
