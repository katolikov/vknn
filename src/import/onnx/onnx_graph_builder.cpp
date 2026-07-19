// See onnx_graph_builder.h. GraphBuilder method bodies: the collect / materialize / SSA-resolve phases.
#include "onnx_graph_builder.h"

namespace vknn {
    namespace onnx {

        void GraphBuilder::build(Reader r) {
            collect(r);
            materializeInitializers();
            ssaResolveNodeIO();
            g.nodes = std::move(nodes);
            dropInitializerInputs();
        }

        bool GraphBuilder::fullyStatic(const Shape &sh) {
            for (int64_t d: sh)
            {
                if (d < 0)
                {
                    return false;
                }
            }
            return true;
        }

        void GraphBuilder::collect(Reader r) {
            uint32_t f, w;
            while (r.tag(f, w))
            {
                switch (f)
                {
                    case kGraphNode: {
                        Node                     n;
                        std::vector<std::string> ni, no;
                        NodeParser::parseNode(r.sub(), n, ni, no, baseDir, &extCache);
                        nodes.push_back(std::move(n));
                        nodeIns.push_back(std::move(ni));
                        nodeOuts.push_back(std::move(no));
                        break;
                    }
                    case kGraphInitializer: {
                        TensorProto tp = TensorProtoParser::parse(r.sub());
                        std::string nm = tp.name;
                        inits.push_back({nm, std::move(tp)});
                        break;
                    }
                    case kGraphInput: {
                        std::string              nm;
                        Shape                    sh;
                        int32_t                  el = 1;
                        std::vector<std::string> params; // per-axis dim_param symbols (empty = concrete)
                        NodeParser::parseValueInfo(r.sub(), nm, sh, el, &params);
                        TensorId id          = g.findOrAdd(nm);
                        g.desc(id).shape     = sh;
                        g.desc(id).dimParams = std::move(params); // retained so inferShapes can bind symbols
                        g.desc(id).dtype     = dtypeFromElem(el);
                        g.desc(id).isInput   = true;
                        g.inputs.push_back(id);
                        break;
                    }
                    case kGraphOutput: {
                        std::string nm;
                        Shape       sh;
                        int32_t     el = 1;
                        NodeParser::parseValueInfo(r.sub(), nm, sh, el);
                        TensorId id = g.findOrAdd(nm);
                        if (fullyStatic(sh))
                        {
                            g.desc(id).shape = sh;
                        }
                        g.desc(id).dtype    = dtypeFromElem(el);
                        g.desc(id).isOutput = true;
                        g.outputs.push_back(id);
                        break;
                    }
                    case kGraphValueInfo: {
                        std::string nm;
                        Shape       sh;
                        int32_t     el = 1;
                        NodeParser::parseValueInfo(r.sub(), nm, sh, el);
                        if (!nm.empty() && !sh.empty())
                        {
                            valueInfoShapes[nm] = sh; // applied to the matching node output in ssaResolveNodeIO
                        }
                        break;
                    }
                    default:
                        r.skip(w);
                        break;
                }
            }
        }

        void GraphBuilder::materializeInitializers() {
            for (auto &pi: inits)
            {
                TensorId id     = g.findOrAdd(pi.name);
                auto    &d      = g.desc(id);
                d.isInitializer = true;
                d.shape         = pi.tp.dims;
                int64_t n       = 1;
                for (auto x: pi.tp.dims)
                {
                    n *= x;
                }
                if (pi.tp.dims.empty())
                {
                    n = 1;
                }
                TensorProtoParser::resolveExternal(baseDir, pi.tp, extCache); // pull EXTERNAL weights from the sibling data file
                HostBuffer hb;
                if (isType(pi.tp.dataType, OnnxType::Int64))
                {
                    d.dtype = DType::Int64;
                    TensorProtoParser::fillHostI64(pi.tp, hb, n);
                } else if (isType(pi.tp.dataType, OnnxType::Int8) || isType(pi.tp.dataType, OnnxType::Uint8))
                {
                    // A pre-quantized int8/uint8 initializer (a QDQ model's weights, a MatMulNBits packed
                    // int4 payload) stays in NATIVE 1-byte host storage instead of widening to fp32 -- an
                    // 8B int4 model's ~4.3 GB of packed weights would otherwise materialize as ~17 GB and
                    // exhaust host RAM at import. The descriptor still records INT8/UINT8 so the dequantize
                    // pass recovers the quantize dtype's saturation range ([-128,127] / [0,255]) from a
                    // zero_point tensor; initFloats decodes the lanes to fp32 on demand, so every reader
                    // still sees integer-valued fp32. (BOOL, which also decodes to UInt8 storage, is
                    // normalized to 0/1 fp32 through fillHostFloat below and is never a quant parameter.)
                    d.dtype = isType(pi.tp.dataType, OnnxType::Int8) ? DType::Int8 : DType::UInt8;
                    TensorProtoParser::fillHostBytes(pi.tp, hb, n, d.dtype);
                } else if (isType(pi.tp.dataType, OnnxType::Double))
                {
                    // A DOUBLE initializer keeps native 8-byte fp64 host storage, never narrowed to fp32,
                    // so it round-trips through the .vxm and a fp64 op reads it in double. initFloats
                    // decodes the lanes to fp32 for the fp32 compute path.
                    d.dtype = DType::Float64;
                    TensorProtoParser::fillHostDouble(pi.tp, hb, n);
                } else
                {
                    // FLOAT / FLOAT16 (and INT32 / BOOL) materialize to fp32 host storage; a plain fp32
                    // read through initFloats recovers the value.
                    d.dtype = DType::Float32;
                    TensorProtoParser::fillHostFloat(pi.tp, hb, n);
                }
                g.initializers[id] = std::move(hb);
            }
        }

        void GraphBuilder::ssaResolveNodeIO() {
            std::unordered_map<std::string, int> producerCount;
            for (const auto &outs: nodeOuts)
            {
                for (const std::string &s: outs)
                {
                    if (!s.empty())
                    {
                        ++producerCount[s];
                    }
                }
            }
            size_t reused = 0;
            for (const auto &kv: producerCount)
            {
                if (kv.second > 1)
                {
                    ++reused;
                }
            }
            if (reused > 0)
            {
                VKNN_WARN << "ONNX graph is not SSA: " << reused << " tensor name(s) have multiple producers "
                          << "(un-deduped trace export). Inputs bind to the nearest preceding producer; "
                          << "value_info shape hints for reused names are ignored.";
            }
            std::unordered_map<std::string, TensorId> latest;
            for (TensorId id: g.inputs)
            {
                latest[g.desc(id).name] = id;
            }
            for (auto &pi: inits)
            {
                latest[pi.name] = g.find(pi.name);
            }
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                for (const std::string &s: nodeIns[i])
                {
                    if (s.empty())
                    {
                        nodes[i].inputs.push_back(kNoTensor);
                        continue;
                    }
                    auto it = latest.find(s);
                    nodes[i].inputs.push_back(it != latest.end() ? it->second : g.findOrAdd(s));
                }
                for (const std::string &s: nodeOuts[i])
                {
                    if (s.empty())
                    {
                        nodes[i].outputs.push_back(kNoTensor);
                        continue;
                    }
                    TensorDesc d;
                    d.name   = s;
                    auto vit = valueInfoShapes.find(s);
                    if (vit != valueInfoShapes.end() && fullyStatic(vit->second) && producerCount[s] == 1)
                    {
                        d.shape = vit->second; // carry the value_info shape hint onto this node output
                    }
                    TensorId id = g.addTensor(std::move(d)); // fresh id; tensorByName[s] -> id (last def wins)
                    latest[s]   = id;
                    nodes[i].outputs.push_back(id);
                }
                if (nodes[i].name.empty())
                {
                    nodes[i].name = std::string(opTypeName(nodes[i].type)) + "_" + std::to_string(i);
                }
            }
            // A declared output name may have been produced by several nodes; point g.outputs at the
            // final producer and carry the declared shape/dtype onto it.
            for (TensorId &oid: g.outputs)
            {
                auto it = latest.find(g.desc(oid).name);
                if (it != latest.end() && it->second != oid)
                {
                    Shape declShape             = g.desc(oid).shape;
                    DType declDtype             = g.desc(oid).dtype;
                    g.desc(it->second).isOutput = true;
                    if (fullyStatic(declShape))
                    {
                        g.desc(it->second).shape = declShape;
                    }
                    g.desc(it->second).dtype = declDtype;
                    oid                      = it->second;
                }
            }
        }

        void GraphBuilder::dropInitializerInputs() {
            std::vector<TensorId> realInputs;
            for (TensorId id: g.inputs)
            {
                if (!g.isInitializer(id))
                {
                    realInputs.push_back(id);
                }
            }
            g.inputs = realInputs;
        }

    } // namespace onnx
} // namespace vknn
