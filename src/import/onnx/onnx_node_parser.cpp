// See onnx_node_parser.h. NodeParser method bodies: NodeProto / AttributeProto / ValueInfoProto wire decode.
#include "onnx_node_parser.h"

namespace vknn {
    namespace onnx {

        void NodeParser::parseAttr(Reader r, Node &node, const std::string &baseDir,
                                   std::map<std::string, std::vector<uint8_t>> *extCache) {
            std::string name;
            Attr        a;
            uint32_t    f, w;
            TensorProto tp;
            bool        hasTp = false;
            while (r.tag(f, w))
            {
                switch (f)
                {
                    case kAttrName:
                        name = r.str();
                        break;
                    case kAttrFloat: {
                        uint32_t b = r.fixed32();
                        std::memcpy(&a.f, &b, 4);
                        a.kind = Attr::Float;
                        break;
                    }
                    case kAttrInt:
                        a.i    = (int64_t) r.varint();
                        a.kind = Attr::Int;
                        break;
                    case kAttrString: {
                        auto b = r.bytes();
                        a.str.assign((const char *) b.data(), b.size());
                        a.kind = Attr::String;
                        break;
                    }
                    case kAttrFloats: {
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
                    case kAttrInts: {
                        // `ints` is a packed repeated field: exporters may emit it length-delimited
                        // (one blob of back-to-back varints) or, for a single element, as a bare
                        // varint. Handle both so an attribute with one int is not dropped.
                        if (w == kWireBytes)
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
                    case kAttrTensor: {
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

        void NodeParser::parseValueInfo(Reader r, std::string &name, Shape &shape, int32_t &elem,
                                        std::vector<std::string> *dimParams) {
            uint32_t f, w;
            while (r.tag(f, w))
            {
                if (f == kValueInfoName)
                {
                    name = r.str();
                } else if (f == kValueInfoType)
                {
                    Reader   tp = r.sub();
                    uint32_t f2, w2;
                    while (tp.tag(f2, w2))
                    {
                        if (f2 == kTypeTensorType)
                        {
                            Reader   tt = tp.sub();
                            uint32_t f3, w3;
                            while (tt.tag(f3, w3))
                            {
                                if (f3 == kTensorTypeElemType)
                                {
                                    elem = (int32_t) tt.varint();
                                } else if (f3 == kTensorTypeShape)
                                {
                                    Reader   sh = tt.sub();
                                    uint32_t f4, w4;
                                    while (sh.tag(f4, w4))
                                    {
                                        if (f4 == kShapeDim)
                                        {
                                            Reader      dim = sh.sub();
                                            uint32_t    f5, w5;
                                            int64_t     val = -1;
                                            std::string param; // dim_param symbol/expression, empty when concrete
                                            while (dim.tag(f5, w5))
                                            {
                                                if (f5 == kDimValue)
                                                {
                                                    val = (int64_t) dim.varint();
                                                } else if (f5 == kDimParam)
                                                {
                                                    param = dim.str(); // retain the symbol name/expression
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

        void NodeParser::parseNode(Reader r, Node &node, std::vector<std::string> &ins, std::vector<std::string> &outs,
                                   const std::string &baseDir,
                                   std::map<std::string, std::vector<uint8_t>> *extCache) {
            uint32_t    f, w;
            std::string opType;
            while (r.tag(f, w))
            {
                switch (f)
                {
                    case kNodeInput:
                        ins.push_back(r.str());
                        break;
                    case kNodeOutput:
                        outs.push_back(r.str());
                        break;
                    case kNodeName:
                        node.name = r.str();
                        break;
                    case kNodeOpType:
                        opType = r.str();
                        break;
                    case kNodeAttribute:
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

    } // namespace onnx
} // namespace vknn
