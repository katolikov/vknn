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
                            TensorId id         = g.findOrAdd(nm);
                            g.desc(id).shape    = sh;
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
                        d.dtype = DType::Float32; // FLOAT / FLOAT16 / DOUBLE all materialize to fp32 storage
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
            void ssaResolveNodeIO() {
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
                        if (vit != valueInfoShapes.end())
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
                        g.desc(it->second).shape    = declShape;
                        g.desc(it->second).dtype    = declDtype;
                        oid                         = it->second;
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
