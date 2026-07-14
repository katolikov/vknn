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
                                  std::map<std::string, std::vector<uint8_t>> *extCache = nullptr) {
                std::string name;
                Attr        a;
                uint32_t    f, w;
                TensorProto tp;
                bool        hasTp = false;
                while (r.tag(f, w))
                {
                    switch (f)
                    {
                        case 1:
                            name = r.str();
                            break;
                        case 2: {
                            uint32_t b = r.fixed32();
                            std::memcpy(&a.f, &b, 4);
                            a.kind = Attr::Float;
                            break;
                        }
                        case 3:
                            a.i    = (int64_t) r.varint();
                            a.kind = Attr::Int;
                            break;
                        case 4: {
                            auto b = r.bytes();
                            a.str.assign((const char *) b.data(), b.size());
                            a.kind = Attr::String;
                            break;
                        }
                        case 7: {
                            Reader s = r.sub();
                            while (!s.eof())
                            {
                                uint32_t b = s.fixed32();
                                float    fv;
                                std::memcpy(&fv, &b, 4);
                                a.floats.push_back(fv);
                            }
                            a.kind = Attr::Floats;
                            break;
                        }
                        case 8: {
                            // `ints` is a packed repeated field: exporters may emit it length-delimited
                            // (wire type 2 = one blob of back-to-back varints) or, for a single element,
                            // as a bare varint. Handle both so an attribute with one int is not dropped.
                            if (w == 2)
                            {
                                Reader s = r.sub();
                                while (!s.eof())
                                {
                                    a.ints.push_back((int64_t) s.varint());
                                }
                            } else
                            {
                                a.ints.push_back((int64_t) r.varint());
                            }
                            a.kind = Attr::Ints;
                            break;
                        }
                        case 5: {
                            tp    = TensorProtoParser::parse(r.sub());
                            hasTp = true;
                            break;
                        }
                        default:
                            r.skip(w);
                            break;
                    }
                }
                if (hasTp)
                {
                    // Pull an EXTERNAL attribute payload (data_location==1) out of the sibling data file into
                    // tp.raw, so the raw_data decode paths below see real bytes instead of the empty inline
                    // payload. No-op for an inline tensor or when no external cache is supplied.
                    if (extCache)
                    {
                        TensorProtoParser::resolveExternal(baseDir, tp, *extCache);
                    }
                    // A tensor-valued attribute (Constant.value, ConstantOfShape, etc.) is flattened into
                    // the same numeric attribute storage as inline floats/ints: int64 tensors -> `ints`,
                    // every other numeric dtype -> `floats`. `n` is the element count the decode loops
                    // read, derived from dims (or the payload length for a rank-0 scalar with empty dims).
                    a.kind    = Attr::Floats;
                    a.shape   = tp.dims; // keep dims so a Constant node emits its true shape (e.g. anchor grids)
                    int64_t n = 1;
                    for (auto d: tp.dims)
                    {
                        n *= d;
                    }
                    if (tp.dims.empty())
                    {
                        n = std::max<int64_t>(1, (int64_t) std::max({tp.floatData.size(), tp.int32Data.size(), tp.int64Data.size()}));
                    }
                    if (!tp.int64Data.empty() || isType(tp.dataType, OnnxType::Int64))
                    {
                        a.kind = Attr::Ints;
                        if (!tp.int64Data.empty())
                        {
                            a.ints = tp.int64Data;
                        } else if (!tp.raw.empty())
                        {
                            const int64_t *s     = (const int64_t *) tp.raw.data();
                            int64_t        avail = (int64_t) (tp.raw.size() / 8);
                            for (int64_t i = 0; i < n && i < avail; ++i)
                            {
                                a.ints.push_back(s[i]);
                            }
                        }
                    } else
                    {
                        // Every non-int64 payload decodes through the same materializer as graph
                        // initializers (fp32 widening by dtype from raw_data or the typed
                        // float_data / int32_data arrays; truncated payloads leave a zero tail).
                        HostBuffer hb;
                        TensorProtoParser::fillHostFloat(tp, hb, n);
                        // Size the range from the buffer's real length, not the unvalidated `n`: a
                        // crafted tensor attribute with a negative dims product yields n < 0, and
                        // hb.f32() + n would form an inverted (last < first) range -> length_error.
                        // fillHostFloat already sized hb to max(n,0) fp32 elements.
                        a.floats.assign(hb.f32(), hb.f32() + hb.bytes.size() / sizeof(float));
                    }
                }
                node.attr.map[name] = a;
            }

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
                                       std::vector<std::string> *dimParams = nullptr) {
                uint32_t f, w;
                while (r.tag(f, w))
                {
                    if (f == 1)
                    {
                        name = r.str();
                    } else if (f == 2)
                    { // TypeProto
                        Reader   tp = r.sub();
                        uint32_t f2, w2;
                        while (tp.tag(f2, w2))
                        {
                            if (f2 == 1)
                            { // tensor_type
                                Reader   tt = tp.sub();
                                uint32_t f3, w3;
                                while (tt.tag(f3, w3))
                                {
                                    if (f3 == 1)
                                    {
                                        elem = (int32_t) tt.varint();
                                    } else if (f3 == 2)
                                    { // shape
                                        Reader   sh = tt.sub();
                                        uint32_t f4, w4;
                                        while (sh.tag(f4, w4))
                                        {
                                            if (f4 == 1)
                                            { // dim
                                                Reader      dim = sh.sub();
                                                uint32_t    f5, w5;
                                                int64_t     val = -1;
                                                std::string param; // dim_param symbol/expression, empty when concrete
                                                while (dim.tag(f5, w5))
                                                {
                                                    if (f5 == 1)
                                                    {
                                                        val = (int64_t) dim.varint(); // dim_value
                                                    } else if (f5 == 2)
                                                    {
                                                        param = dim.str(); // dim_param: retain the symbol name/expression
                                                        val   = -1;
                                                    } else
                                                    {
                                                        dim.skip(w5);
                                                    }
                                                }
                                                shape.push_back(val);
                                                if (dimParams)
                                                {
                                                    dimParams->push_back(std::move(param));
                                                }
                                            } else
                                            {
                                                sh.skip(w4);
                                            }
                                        }
                                    } else
                                    {
                                        tt.skip(w3);
                                    }
                                }
                            } else
                            {
                                tp.skip(w2);
                            }
                        }
                    } else
                    {
                        r.skip(w);
                    }
                }
            }

            // ----------------------------- NodeProto -----------------------------
            // fields: 1=input(string repeated), 2=output(string repeated), 3=name, 4=op_type, 5=attribute,
            // 7=domain. Returns the raw input/output tensor NAMES (in `ins`/`outs`) instead of resolving them
            // to ids here; GraphBuilder resolves them in its SSA pass so a trace that REUSES a tensor name
            // (two nodes both writing "Cast_output_0" — common in un-deduped PyTorch exports) does not
            // collapse onto one TensorId.
            static void parseNode(Reader r, Node &node, std::vector<std::string> &ins, std::vector<std::string> &outs,
                                  const std::string &baseDir = std::string(),
                                  std::map<std::string, std::vector<uint8_t>> *extCache = nullptr) {
                uint32_t    f, w;
                std::string opType;
                while (r.tag(f, w))
                {
                    switch (f)
                    {
                        case 1:
                            ins.push_back(r.str());
                            break;
                        case 2:
                            outs.push_back(r.str());
                            break;
                        case 3:
                            node.name = r.str();
                            break;
                        case 4:
                            opType = r.str();
                            break;
                        case 5:
                            parseAttr(r.sub(), node, baseDir, extCache);
                            break;
                        default:
                            r.skip(w);
                            break;
                    }
                }
                // Opset-9/10 Upsample carries `scales` at input 1, but its OpType::Resize consumers
                // (the shape rule and both kernels) index the Resize convention X, roi, scales,
                // sizes. Insert the absent-roi slot so scales lands at input 2 and every consumer
                // reads one layout; an empty name resolves to kNoTensor in the graph builder.
                if (opType == "Upsample" && ins.size() == 2)
                {
                    ins.insert(ins.begin() + 1, std::string());
                }
                // Collapse the ONNX op name into a coarse OpType plus, for the multiplexed families
                // (Unary/Binary/Reduce), a `subOp` selecting the concrete variant. Activation ops carry
                // their curve parameters in attributes, applied here with the ONNX-specified defaults so
                // an exporter that omits them still lowers to the correct activation.
                node.type = opTypeFromOnnx(opType);
                if (node.type == OpType::Unary)
                {
                    UnaryType u = unaryFromOnnx(opType);
                    node.subOp  = (int32_t) u;
                    // params (defaults per ONNX): LeakyRelu alpha=0.01, Elu alpha=1.0, HardSigmoid alpha,beta
                    if (u == UnaryType::LeakyRelu)
                    {
                        node.actLo = node.attr.getf("alpha", 0.01f);
                    } else if (u == UnaryType::Elu)
                    {
                        node.actLo = node.attr.getf("alpha", 1.0f);
                    } else if (u == UnaryType::HardSigmoid)
                    {
                        node.actLo = node.attr.getf("alpha", 0.2f);
                        node.actHi = node.attr.getf("beta", 0.5f);
                    }
                } else if (node.type == OpType::Binary)
                {
                    node.subOp = (int32_t) binaryFromOnnx(opType);
                } else if (node.type == OpType::Reduce)
                { node.subOp = (int32_t) reduceFromOnnx(opType); }
                if (node.type == OpType::Unknown)
                {
                    VKNN_WARN << "unknown ONNX op '" << opType << "' (node " << node.name << ")";
                } else if (opTypeIsQuantized(node.type))
                {
                    // Quantized ops import as their own OpType but have no kernel; only the
                    // import-time dequantize lowering makes a graph that carries them runnable.
                    VKNN_WARN << "quantized ONNX op '" << opType << "' (node " << node.name
                              << ") - runs only via the dequantize lowering";
                }
            }
        };

    } // namespace onnx
} // namespace vknn
