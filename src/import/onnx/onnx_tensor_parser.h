// Reads a TensorProto off the wire and materializes it into a HostBuffer (raw_data / typed data /
// external data), decoding FLOAT / FLOAT16 / DOUBLE / INT64 / INT32 / INT8 / UINT8 / BOOL to the
// fp32 or int64 compute storage.
#pragma once
#include "onnx_reader.h"
#include "onnx_types.h"
#include "vknn/graph.h"
#include "vknn/logging.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace vknn {
    namespace onnx {

        // Decodes a single ONNX TensorProto (field numbers per onnx_types.h) and, on demand, resolves
        // its external-data reference and materializes its payload into a HostBuffer. A repeated numeric
        // field arrives either length-delimited (kWireBytes, protobuf's packed encoding: all
        // values concatenated inside one sub-message) or as one value per tag; every such field below
        // branches on `w` to handle both, since exporters differ on which they emit.
        class TensorProtoParser {
          public:
            static TensorProto parse(Reader r);

            // Resolve an EXTERNAL tensor (data_location==1) by reading its byte range from the sibling data
            // file into t.raw, so the normal raw_data paths handle it. The data file is read once and cached
            // (one .onnx.data backs every weight). No-op if the tensor isn't external or already inline.
            static void resolveExternal(const std::string &baseDir, TensorProto &t, std::map<std::string, std::vector<uint8_t>> &cache);

            // Materialize a TensorProto into a float32 HostBuffer, decoding whichever payload the proto
            // carries: raw_data (FLOAT copied verbatim; FLOAT16 / DOUBLE / INT64 / INT32 / INT8 / UINT8
            // / BOOL converted per element) or the typed float_data / int64_data / int32_data arrays.
            // `elems` is the element count implied by the tensor's shape; every copy is clamped to what
            // the payload actually holds (`min` / `i < avail`), so a truncated or shape-mismatched proto
            // leaves the tail zero rather than reading out of bounds.
            static void fillHostFloat(const TensorProto &t, HostBuffer &hb, int64_t elems);

            // Materialize an INT8 / UINT8 TensorProto into NATIVE 1-byte-per-element host storage, without
            // the fp32 widening fillHostFloat does. A pre-quantized weight (a QDQ model's int8 weights, a
            // MatMulNBits packed int4 payload) keeps its on-disk size in host memory -- an 8B int4 model's
            // ~4.3 GB of packed weights would otherwise quadruple to ~17 GB and exhaust host RAM at import.
            // initFloats() decodes these lanes back to fp32 on demand, so every downstream reader still sees
            // integer-valued fp32. Copies are clamped to the payload the same way as fillHostFloat, leaving a
            // truncated or shape-mismatched tail zero.
            static void fillHostBytes(const TensorProto &t, HostBuffer &hb, int64_t elems, DType dt);

            // Materialize as int64 (shape / index tensors that must stay exact). Only the two lossless
            // int64 sources are honored: raw_data of dtype INT64, or the typed int64_data array; any
            // other dtype leaves the buffer zero-filled. Copies are clamped to the payload the same way
            // as fillHostFloat.
            static void fillHostI64(const TensorProto &t, HostBuffer &hb, int64_t elems);

            // Materialize a DOUBLE TensorProto into native 8-byte fp64 host storage: raw_data DOUBLE is
            // copied verbatim, the typed double_data array element-wise, with no narrowing to fp32.
            // initFloats() decodes the lanes to fp32 for the fp32 compute path; a fp64 op and the
            // declared-dtype output readback read them through f64(). Copies are clamped to the payload
            // as in fillHostFloat, leaving a truncated or shape-mismatched tail zero.
            static void fillHostDouble(const TensorProto &t, HostBuffer &hb, int64_t elems);
        };

    } // namespace onnx
} // namespace vknn
