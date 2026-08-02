// Naming shared by the layer-dump writers and the node-order index they point at.
#pragma once
#include <string>

namespace vknn {

    /// File name a dumped tensor takes, without extension.
    ///
    /// A tensor name is whatever the exporter chose, and an ONNX exporter routinely prefixes every
    /// internal tensor with a module path; those separators cannot appear in a file name, so the
    /// dump flattens them. The node-order index has to spell tensors the same way or every name it
    /// prints refers to a file that does not exist -- and a diff walk that matches names then
    /// silently compares only the tensors whose names happened to need no flattening, which on such
    /// a graph is the boundary tensors alone.
    inline std::string layerDumpFileName(std::string tensorName) {
        for (char &c: tensorName)
        {
            if (c == '/' || c == ':')
            {
                c = '_';
            }
        }
        return tensorName;
    }

} // namespace vknn
