// Builds a vknn Graph from a GraphProto in phases: collect() reads the proto (deferring node I/O to raw
// name lists), materializeInitializers() fills weight buffers, ssaResolveNodeIO() gives every node output
// a fresh TensorId (so a trace that reuses tensor names does not alias distinct tensors), and
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

namespace vknn {
    namespace onnx {

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
            std::vector<std::vector<std::string>>       nodeIns, nodeOuts; // raw names, resolved in ssaResolveNodeIO
            std::map<std::string, Shape>                valueInfoShapes;   // value_info hints -> node outputs

          public:
            GraphBuilder(Graph &graph, const std::string &dir): g(graph), baseDir(dir) {
            }

            void build(Reader r) {
                collect(r);
                materializeInitializers();
                ssaResolveNodeIO();
                g.nodes = std::move(nodes);
                dropInitializerInputs();
            }

          private:
            // A declared shape is only trusted when fully static. dim_param (symbolic) dims parse
            // to -1; storing them as a desc would read as "resolved" and poison downstream
            // inference (a Reshape 0-copy of a -1, a Slice clamp against -1 -> 0, then a Shape()
            // fold freezes the lie). Graph inputs keep -1 dims: runStandardPasses substitutes the
            // static batch there.
            static bool fullyStatic(const Shape &sh) {
                for (int64_t d: sh)
                {
                    if (d < 0)
                    {
                        return false;
                    }
                }
                return true;
            }

            // Reads the GraphProto, dispatching on its protobuf field number: 1 = node (deferred to raw
            // name lists here; I/O is bound in ssaResolveNodeIO), 5 = initializer (deferred to inits;
            // weights are filled in materializeInitializers), 11 = graph input, 12 = graph output,
            // 13 = value_info (a shape hint, applied to its node output in ssaResolveNodeIO). Any other
            // field is skipped. Inputs/outputs are registered by name (findOrAdd); a symbolic output
            // shape is dropped (fullyStatic), an input keeps its -1 dims for the later batch substitution.
            void collect(Reader r) {
                uint32_t f, w;
                while (r.tag(f, w))
                {
                    switch (f)
                    {
                        case 1: {
                            Node                     n;
                            std::vector<std::string> ni, no;
                            NodeParser::parseNode(r.sub(), n, ni, no);
                            nodes.push_back(std::move(n));
                            nodeIns.push_back(std::move(ni));
                            nodeOuts.push_back(std::move(no));
                            break;
                        }
                        case 5: {
                            TensorProto tp = TensorProtoParser::parse(r.sub());
                            std::string nm = tp.name;
                            inits.push_back({nm, std::move(tp)});
                            break;
                        }
                        case 11: {
                            std::string nm;
                            Shape       sh;
                            int32_t     el = 1;
                            NodeParser::parseValueInfo(r.sub(), nm, sh, el);
                            TensorId id        = g.findOrAdd(nm);
                            g.desc(id).shape   = sh;
                            g.desc(id).dtype   = dtypeFromElem(el);
                            g.desc(id).isInput = true;
                            g.inputs.push_back(id);
                            break;
                        }
                        case 12: {
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
                        case 13: {
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

            // Turns each initializer collected by collect() into a graph tensor with a host weight buffer.
            // Runs before ssaResolveNodeIO so initializer names resolve to real ids when node inputs bind.
            // External initializers (data held in a sibling file) are pulled in first via resolveExternal.
            // Storage dtype is narrowed to two host representations: Int64 stays Int64; every float variant
            // (FLOAT / FLOAT16 / DOUBLE) materializes to fp32. A scalar initializer has empty dims, so its
            // element count is forced to 1.
            void materializeInitializers() {
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
                    } else
                    {
                        // Payload materializes to fp32 host storage regardless (FLOAT / FLOAT16 / DOUBLE and
                        // the widened integer narrows all read through initFloats, which treats any non-fp16
                        // dtype as fp32 bytes). The descriptor dtype, though, records INT8/UINT8 for a
                        // quantized initializer so the dequantize pass can recover the quantize dtype's
                        // saturation range ([-128,127] / [0,255]) from a zero_point tensor -- lost otherwise,
                        // since the range decides the clamp that a QDQ collapse must preserve. BOOL (which
                        // also decodes to UInt8 storage) is deliberately not stamped: it is normalized to
                        // 0/1 fp32 and is never a quantization parameter.
                        d.dtype = isType(pi.tp.dataType, OnnxType::Int8)    ? DType::Int8
                                : isType(pi.tp.dataType, OnnxType::Uint8)   ? DType::UInt8
                                                                            : DType::Float32;
                        TensorProtoParser::fillHostFloat(pi.tp, hb, n);
                    }
                    g.initializers[id] = std::move(hb);
                }
            }

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
            void ssaResolveNodeIO() {
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

            // ONNX lists initializers in the graph input list too; drop them.
            void dropInitializerInputs() {
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
        };

    } // namespace onnx
} // namespace vknn
