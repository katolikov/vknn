// Parses NodeProto / AttributeProto / ValueInfoProto off the wire. Node I/O is returned as raw tensor
// NAMES (resolved later by GraphBuilder's SSA pass), and attribute tensors are materialized to fp32/int
// attribute values decoding by dtype.
#pragma once
#include "onnx_reader.h"
#include "onnx_tensor_parser.h"
#include "onnx_types.h"
#include "vknn/graph.h"
#include "vknn/logging.h"
#include "vknn/op.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace vknn {
    namespace onnx {

        class NodeParser {
          public:
            // ----------------------------- AttributeProto -----------------------------
            // fields: 1=name, 2=f(float32), 3=i(int64), 4=s(bytes), 7=floats(packed), 8=ints(packed),
            // 5=t(TensorProto), 20=type(int32)
            // A tensor-valued attribute (Constant.value, ConstantOfShape.value) may hold its payload in a
            // sibling external-data file exactly like a graph initializer; @p baseDir and @p extCache let it
            // resolve that reference before materializing, so an external Constant is not read as all-zeros.
            static void parseAttr(Reader r, Node &node, const std::string &baseDir = std::string(),
                                  std::map<std::string, std::vector<uint8_t>> *extCache = nullptr);

            // ----------------------------- ValueInfoProto -----------------------------
            // field 1 = name, field 2 = type(TypeProto); TypeProto field1=tensor_type;
            // Tensor field1=elem_type(int32), field2=shape(TensorShapeProto);
            // TensorShapeProto field1=dim(repeated Dimension); Dimension field1=dim_value(int64).
            // A symbolic dimension (Dimension.dim_param, field 2) has no static extent and is recorded
            // as -1, so downstream shape inference treats it as the unknown/dynamic axis. When @p dimParams
            // is non-null it is filled parallel to @p shape: the symbol/expression string of a symbolic
            // axis (the raw dim_param, e.g. "past_sequence_length" or "past_sequence_length + sequence_length")
            // and an empty string for a concrete axis, so a dynamic dim can later be resolved by binding its
            // symbol instead of a full per-tensor shape.
            static void parseValueInfo(Reader r, std::string &name, Shape &shape, int32_t &elem,
                                       std::vector<std::string> *dimParams = nullptr);

            // ----------------------------- NodeProto -----------------------------
            // fields: 1=input(string repeated), 2=output(string repeated), 3=name, 4=op_type, 5=attribute,
            // 7=domain. Returns the raw input/output tensor NAMES (in `ins`/`outs`) instead of resolving them
            // to ids here; GraphBuilder resolves them in its SSA pass so a trace that REUSES a tensor name
            // (two nodes both writing "Cast_output_0" — common in un-deduped PyTorch exports) does not
            // collapse onto one TensorId.
            static void parseNode(Reader r, Node &node, std::vector<std::string> &ins, std::vector<std::string> &outs,
                                  const std::string &baseDir = std::string(),
                                  std::map<std::string, std::vector<uint8_t>> *extCache = nullptr);
        };

    } // namespace onnx
} // namespace vknn
