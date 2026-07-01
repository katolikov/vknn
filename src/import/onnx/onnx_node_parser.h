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
#include <string>
#include <vector>

namespace vknn {
    namespace onnx {

        class NodeParser {
          public:
            // ----------------------------- AttributeProto -----------------------------
            // fields: 1=name, 2=f(float32), 3=i(int64), 4=s(bytes), 7=floats(packed), 8=ints(packed),
            // 5=t(TensorProto), 20=type(int32)
            static void parseAttr(Reader r, Node &node) {
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
                    // store as a float-ints attribute when it's a small shape/scalar constant
                    a.kind    = Attr::Floats;
                    a.shape   = tp.dims; // keep dims so a Constant node emits its true shape (e.g. anchor grids)
                    int64_t n = 1;
                    for (auto d: tp.dims)
                    {
                        n *= d;
                    }
                    if (tp.dims.empty())
                    {
                        n = std::max<int64_t>(1, (int64_t) std::max(tp.floatData.size(), tp.int64Data.size()));
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
                        if (!tp.floatData.empty())
                        {
                            a.floats = tp.floatData;
                        } else if (!tp.raw.empty())
                        {
                            // Decode by the tensor's dtype: FLOAT16 raw is 2 bytes/elem (reading it as fp32 would
                            // over-read 2x and fault); FLOAT raw is 4 bytes/elem. Bounds-checked either way.
                            if (isType(tp.dataType, OnnxType::Float16))
                            {
                                const uint16_t *s     = (const uint16_t *) tp.raw.data();
                                int64_t         avail = (int64_t) (tp.raw.size() / 2);
                                for (int64_t i = 0; i < n && i < avail; ++i)
                                {
                                    a.floats.push_back(halfToFloat(s[i]));
                                }
                            } else if (isType(tp.dataType, OnnxType::Double))
                            {
                                const double *s     = (const double *) tp.raw.data();
                                int64_t       avail = (int64_t) (tp.raw.size() / 8);
                                for (int64_t i = 0; i < n && i < avail; ++i)
                                {
                                    a.floats.push_back((float) s[i]);
                                }
                            } else
                            {
                                const float *s     = (const float *) tp.raw.data();
                                int64_t      avail = (int64_t) (tp.raw.size() / 4);
                                for (int64_t i = 0; i < n && i < avail; ++i)
                                {
                                    a.floats.push_back(s[i]);
                                }
                            }
                        }
                    }
                }
                node.attr.map[name] = a;
            }

            // ----------------------------- ValueInfoProto -----------------------------
            // field 1 = name, field 2 = type(TypeProto); TypeProto field1=tensor_type;
            // Tensor field1=elem_type(int32), field2=shape(TensorShapeProto);
            // TensorShapeProto field1=dim(repeated Dimension); Dimension field1=dim_value(int64).
            static void parseValueInfo(Reader r, std::string &name, Shape &shape, int32_t &elem) {
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
                                                Reader   dim = sh.sub();
                                                uint32_t f5, w5;
                                                int64_t  val = -1;
                                                while (dim.tag(f5, w5))
                                                {
                                                    if (f5 == 1)
                                                    {
                                                        val = (int64_t) dim.varint(); // dim_value
                                                    } else if (f5 == 2)
                                                    {
                                                        dim.str();
                                                        val = -1;
                                                    } // dim_param (dynamic)
                                                    else
                                                    {
                                                        dim.skip(w5);
                                                    }
                                                }
                                                shape.push_back(val);
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
            static void parseNode(Reader r, Node &node, std::vector<std::string> &ins, std::vector<std::string> &outs) {
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
                            parseAttr(r.sub(), node);
                            break;
                        default:
                            r.skip(w);
                            break;
                    }
                }
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
                }
            }
        };

    } // namespace onnx
} // namespace vknn
