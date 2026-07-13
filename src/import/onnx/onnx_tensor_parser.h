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
        // field arrives either length-delimited (wire type `w == 2`, protobuf's packed encoding: all
        // values concatenated inside one sub-message) or as one value per tag; every such field below
        // branches on `w` to handle both, since exporters differ on which they emit.
        class TensorProtoParser {
          public:
            static TensorProto parse(Reader r) {
                TensorProto t;
                uint32_t    f, w;
                while (r.tag(f, w))
                {
                    switch (f)
                    {
                        case 1: // dims: packed sub-message of varints, or a single varint
                            if (w == 2)
                            {
                                Reader s = r.sub();
                                while (!s.eof())
                                {
                                    t.dims.push_back((int64_t) s.varint());
                                }
                            } else
                            {
                                t.dims.push_back((int64_t) r.varint());
                            }
                            break;
                        case 2:
                            t.dataType = (int32_t) r.varint();
                            break;
                        case 4: // float_data (packed or single)
                            if (w == 2)
                            {
                                Reader s = r.sub();
                                while (!s.eof())
                                {
                                    uint32_t b = s.fixed32();
                                    float    fv;
                                    std::memcpy(&fv, &b, 4);
                                    t.floatData.push_back(fv);
                                }
                            } else
                            {
                                uint32_t b = r.fixed32();
                                float    fv;
                                std::memcpy(&fv, &b, 4);
                                t.floatData.push_back(fv);
                            }
                            break;
                        case 5: // int32_data (packed or single): typed payload for INT32 and every
                                // narrower type. A negative int32 arrives as a 64-bit sign-extended
                                // varint; truncating the decoded u64 to int32 recovers its value.
                            if (w == 2)
                            {
                                Reader s = r.sub();
                                while (!s.eof())
                                {
                                    t.int32Data.push_back((int32_t) s.varint());
                                }
                            } else
                            {
                                t.int32Data.push_back((int32_t) r.varint());
                            }
                            break;
                        case 7: // int64_data
                            if (w == 2)
                            {
                                Reader s = r.sub();
                                while (!s.eof())
                                {
                                    t.int64Data.push_back((int64_t) s.varint());
                                }
                            } else
                            {
                                t.int64Data.push_back((int64_t) r.varint());
                            }
                            break;
                        case 8:
                            t.name = r.str();
                            break;
                        case 9:
                            t.raw = r.bytes();
                            break;
                        case 13: { // external_data: StringStringEntryProto { 1=key, 2=value }
                            Reader      s = r.sub();
                            std::string key, val;
                            uint32_t    ef, ew;
                            while (s.tag(ef, ew))
                            {
                                if (ef == 1 && ew == 2)
                                {
                                    key = s.str();
                                } else if (ef == 2 && ew == 2)
                                {
                                    val = s.str();
                                } else
                                {
                                    s.skip(ew);
                                }
                            }
                            if (key == "location")
                            {
                                t.extLoc = val;
                            } else if (key == "offset")
                            {
                                t.extOffset = std::strtoll(val.c_str(), nullptr, 10);
                            } else if (key == "length")
                            { t.extLength = std::strtoll(val.c_str(), nullptr, 10); }
                            break;
                        }
                        case 14: // data_location
                            t.dataLocation = (int32_t) r.varint();
                            break;
                        default:
                            r.skip(w);
                            break;
                    }
                }
                return t;
            }

            // Resolve an EXTERNAL tensor (data_location==1) by reading its byte range from the sibling data
            // file into t.raw, so the normal raw_data paths handle it. The data file is read once and cached
            // (one .onnx.data backs every weight). No-op if the tensor isn't external or already inline.
            static void resolveExternal(const std::string &baseDir, TensorProto &t, std::map<std::string, std::vector<uint8_t>> &cache) {
                if (t.dataLocation != 1 || t.extLoc.empty() || !t.raw.empty())
                {
                    return;
                }
                std::string path = baseDir.empty() ? t.extLoc : baseDir + "/" + t.extLoc;
                auto        it   = cache.find(path);
                if (it == cache.end())
                {
                    std::ifstream        f(path, std::ios::binary);
                    std::vector<uint8_t> buf;
                    if (f)
                    {
                        buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                    } else
                    {
                        VKNN_ERROR << "external data file not found: " << path << " (for tensor '" << t.name << "')";
                    }
                    it = cache.emplace(path, std::move(buf)).first;
                }
                const std::vector<uint8_t> &file = it->second;
                const int64_t               fsz  = (int64_t) file.size();
                int64_t                     off  = t.extOffset;
                int64_t                     len  = t.extLength >= 0 ? t.extLength : fsz - off;
                // Compare without forming off+len: a crafted offset/length (both near INT64_MAX) would
                // overflow that sum to a negative value and slip past a `> fsz` test. off <= fsz makes
                // `fsz - off` non-negative, so the length test cannot overflow either.
                if (off < 0 || len < 0 || off > fsz || len > fsz - off)
                {
                    VKNN_ERROR << "external data range [" << off << "," << off + len << ") out of bounds for '" << t.name << "' (file " << file.size() << " bytes)";
                    return;
                }
                t.raw.assign(file.begin() + off, file.begin() + off + len);
            }

            // Materialize a TensorProto into a float32 HostBuffer, decoding whichever payload the proto
            // carries: raw_data (FLOAT copied verbatim; FLOAT16 / DOUBLE / INT64 / INT32 / INT8 / UINT8
            // / BOOL converted per element) or the typed float_data / int64_data / int32_data arrays.
            // `elems` is the element count implied by the tensor's shape; every copy is clamped to what
            // the payload actually holds (`min` / `i < avail`), so a truncated or shape-mismatched proto
            // leaves the tail zero rather than reading out of bounds.
            static void fillHostFloat(const TensorProto &t, HostBuffer &hb, int64_t elems) {
                hb.resizeElems(elems, DType::Float32);
                float *dst = hb.f32();
                if (!t.raw.empty())
                {
                    if (isType(t.dataType, OnnxType::Float))
                    {
                        // Clamp to the destination byte size, not (size_t)elems*4: a negative elems from an
                        // overflowed dims product sizes the buffer to 0 (resizeElems) yet casts to a huge
                        // size_t, so raw bytes would be memcpy'd into an empty destination.
                        std::memcpy(dst, t.raw.data(), std::min<size_t>(t.raw.size(), hb.bytes.size()));
                    } else if (isType(t.dataType, OnnxType::Float16))
                    { // decode to fp32 (2 bytes/elem)
                        const uint16_t *s     = reinterpret_cast<const uint16_t *>(t.raw.data());
                        int64_t         avail = (int64_t) (t.raw.size() / 2);
                        for (int64_t i = 0; i < elems && i < avail; ++i)
                        {
                            dst[i] = halfToFloat(s[i]);
                        }
                    } else if (isType(t.dataType, OnnxType::Double))
                    { // narrow to fp32 (8 bytes/elem)
                        const double *s     = reinterpret_cast<const double *>(t.raw.data());
                        int64_t       avail = (int64_t) (t.raw.size() / 8);
                        for (int64_t i = 0; i < elems && i < avail; ++i)
                        {
                            dst[i] = (float) s[i];
                        }
                    } else if (isType(t.dataType, OnnxType::Int64))
                    {
                        const int64_t *s     = reinterpret_cast<const int64_t *>(t.raw.data());
                        int64_t        avail = (int64_t) (t.raw.size() / 8);
                        for (int64_t i = 0; i < elems && i < avail; ++i)
                        {
                            dst[i] = (float) s[i];
                        }
                    } else if (isType(t.dataType, OnnxType::Int32))
                    { // widen to fp32 (4 bytes/elem)
                        const int32_t *s     = reinterpret_cast<const int32_t *>(t.raw.data());
                        int64_t        avail = (int64_t) (t.raw.size() / 4);
                        for (int64_t i = 0; i < elems && i < avail; ++i)
                        {
                            dst[i] = (float) s[i];
                        }
                    } else if (isType(t.dataType, OnnxType::Int8))
                    { // widen to fp32 (1 byte/elem, signed)
                        const int8_t *s     = reinterpret_cast<const int8_t *>(t.raw.data());
                        int64_t       avail = (int64_t) t.raw.size();
                        for (int64_t i = 0; i < elems && i < avail; ++i)
                        {
                            dst[i] = (float) s[i];
                        }
                    } else if (isType(t.dataType, OnnxType::Uint8))
                    { // widen to fp32 (1 byte/elem, unsigned)
                        const uint8_t *s     = t.raw.data();
                        int64_t        avail = (int64_t) t.raw.size();
                        for (int64_t i = 0; i < elems && i < avail; ++i)
                        {
                            dst[i] = (float) s[i];
                        }
                    } else if (isType(t.dataType, OnnxType::Bool))
                    { // one byte per element, normalized to exactly 0/1
                        const uint8_t *s     = t.raw.data();
                        int64_t        avail = (int64_t) t.raw.size();
                        for (int64_t i = 0; i < elems && i < avail; ++i)
                        {
                            dst[i] = s[i] ? 1.0f : 0.0f;
                        }
                    }
                } else if (!t.floatData.empty())
                {
                    for (int64_t i = 0; i < elems && i < (int64_t) t.floatData.size(); ++i)
                    {
                        dst[i] = t.floatData[i];
                    }
                } else if (!t.int64Data.empty())
                {
                    for (int64_t i = 0; i < elems && i < (int64_t) t.int64Data.size(); ++i)
                    {
                        dst[i] = (float) t.int64Data[i];
                    }
                } else if (!t.int32Data.empty())
                {
                    // int32_data carries INT32 and every narrower type: FLOAT16 rides as raw bit
                    // patterns (decoded through halfToFloat), BOOL normalizes to exactly 0/1, and
                    // the integer types widen numerically.
                    const bool f16 = isType(t.dataType, OnnxType::Float16);
                    const bool bl  = isType(t.dataType, OnnxType::Bool);
                    for (int64_t i = 0; i < elems && i < (int64_t) t.int32Data.size(); ++i)
                    {
                        int32_t v = t.int32Data[i];
                        dst[i]    = f16 ? halfToFloat((uint16_t) v) : bl ? (v ? 1.0f : 0.0f) : (float) v;
                    }
                }
            }

            // Materialize an INT8 / UINT8 TensorProto into NATIVE 1-byte-per-element host storage, without
            // the fp32 widening fillHostFloat does. A pre-quantized weight (a QDQ model's int8 weights, a
            // MatMulNBits packed int4 payload) keeps its on-disk size in host memory -- an 8B int4 model's
            // ~4.3 GB of packed weights would otherwise quadruple to ~17 GB and exhaust host RAM at import.
            // initFloats() decodes these lanes back to fp32 on demand, so every downstream reader still sees
            // integer-valued fp32. Copies are clamped to the payload the same way as fillHostFloat, leaving a
            // truncated or shape-mismatched tail zero.
            static void fillHostBytes(const TensorProto &t, HostBuffer &hb, int64_t elems, DType dt) {
                hb.resizeElems(elems, dt); // 1 byte/elem for INT8 / UINT8
                uint8_t *dst = hb.bytes.data();
                if (!t.raw.empty())
                {
                    std::memcpy(dst, t.raw.data(), std::min<size_t>(t.raw.size(), (size_t) std::max<int64_t>(elems, 0)));
                } else if (!t.int32Data.empty())
                { // a narrow integer type can ride in int32_data (one value per tag); keep its low byte
                    int64_t avail = (int64_t) t.int32Data.size();
                    for (int64_t i = 0; i < elems && i < avail; ++i)
                    {
                        dst[i] = (uint8_t) t.int32Data[i];
                    }
                }
            }

            // Materialize as int64 (shape / index tensors that must stay exact). Only the two lossless
            // int64 sources are honored: raw_data of dtype INT64, or the typed int64_data array; any
            // other dtype leaves the buffer zero-filled. Copies are clamped to the payload the same way
            // as fillHostFloat.
            static void fillHostI64(const TensorProto &t, HostBuffer &hb, int64_t elems) {
                hb.resizeElems(elems, DType::Int64);
                int64_t *dst = hb.i64();
                if (!t.raw.empty() && isType(t.dataType, OnnxType::Int64))
                {
                    // Clamp to the destination byte size (see fillHostFloat): a negative elems must not
                    // widen the copy length past the 0-byte buffer resizeElems produced.
                    std::memcpy(dst, t.raw.data(), std::min<size_t>(t.raw.size(), hb.bytes.size()));
                } else if (!t.int64Data.empty())
                {
                    for (int64_t i = 0; i < elems && i < (int64_t) t.int64Data.size(); ++i)
                    {
                        dst[i] = t.int64Data[i];
                    }
                }
            }
        };

    } // namespace onnx
} // namespace vknn
