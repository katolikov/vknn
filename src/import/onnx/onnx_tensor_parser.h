// Reads a TensorProto off the wire and materializes it into a HostBuffer (raw_data / typed data /
// external data), decoding FLOAT / FLOAT16 / DOUBLE / INT64 to the fp32 or int64 compute storage.
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

        class TensorProtoParser {
          public:
            static TensorProto parse(Reader r) {
                TensorProto t;
                uint32_t    f, w;
                while (r.tag(f, w))
                {
                    switch (f)
                    {
                        case 1: // dims
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
                int64_t                     off  = t.extOffset;
                int64_t                     len  = t.extLength >= 0 ? t.extLength : (int64_t) file.size() - off;
                if (off < 0 || len < 0 || off + len > (int64_t) file.size())
                {
                    VKNN_ERROR << "external data range [" << off << "," << off + len << ") out of bounds for '" << t.name << "' (file " << file.size() << " bytes)";
                    return;
                }
                t.raw.assign(file.begin() + off, file.begin() + off + len);
            }

            // Materialize a TensorProto into a float32 HostBuffer (raw_data or typed data).
            static void fillHostFloat(const TensorProto &t, HostBuffer &hb, int64_t elems) {
                hb.resizeElems(elems, DType::Float32);
                float *dst = hb.f32();
                if (!t.raw.empty())
                {
                    if (isType(t.dataType, OnnxType::Float))
                    {
                        std::memcpy(dst, t.raw.data(), std::min<size_t>(t.raw.size(), (size_t) elems * 4));
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
                }
            }

            // Materialize as int64 (for shape tensors).
            static void fillHostI64(const TensorProto &t, HostBuffer &hb, int64_t elems) {
                hb.resizeElems(elems, DType::Int64);
                int64_t *dst = hb.i64();
                if (!t.raw.empty() && isType(t.dataType, OnnxType::Int64))
                {
                    std::memcpy(dst, t.raw.data(), std::min<size_t>(t.raw.size(), (size_t) elems * 8));
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
