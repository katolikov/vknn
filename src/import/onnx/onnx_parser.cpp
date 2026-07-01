// ONNX (protobuf) importer entry point. Parses the protobuf wire format directly (no protobuf library /
// generated code). The pieces each live in their own header: the wire reader (onnx_reader.h), the dtype
// enum + TensorProto holder (onnx_types.h), the tensor materializer (onnx_tensor_parser.h), the node /
// attribute / value-info parser (onnx_node_parser.h), and the graph builder (onnx_graph_builder.h). This
// file just wires them together into importOnnx().
#include "onnx_graph_builder.h"
#include "onnx_reader.h"
#include "vknn/graph.h"
#include "vknn/logging.h"
#include <fstream>
#include <string>
#include <vector>

namespace vknn {
    namespace onnx {

        static void parseGraph(Reader r, Graph &g, const std::string &baseDir) {
            GraphBuilder(g, baseDir).build(r);
        }

    } // namespace onnx

    // Public entry point.
    Graph importOnnx(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            throw Error(Status::IoError, "cannot open ONNX file: " + path);
        }
        std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        // Directory the model lives in — external_data locations are relative to it.
        size_t      slash   = path.find_last_of("/\\");
        std::string baseDir = slash == std::string::npos ? std::string() : path.substr(0, slash);
        Graph       g;
        // ModelProto: field 7 = graph (GraphProto). Skip everything else.
        onnx::Reader r(buf.data(), buf.size());
        uint32_t     fld, wire;
        bool         foundGraph = false;
        while (r.tag(fld, wire))
        {
            if (fld == 7 && wire == 2)
            {
                onnx::parseGraph(r.sub(), g, baseDir);
                foundGraph = true;
            } else
            {
                r.skip(wire);
            }
        }
        if (!foundGraph)
        {
            throw Error(Status::InvalidArgument, "no GraphProto in ONNX model");
        }
        g.topoSort();
        VKNN_INFO << "Imported ONNX: " << g.nodes.size() << " nodes, " << g.initializers.size() << " initializers, " << g.inputs.size() << " inputs, "
                  << g.outputs.size() << " outputs";
        return g;
    }

} // namespace vknn
