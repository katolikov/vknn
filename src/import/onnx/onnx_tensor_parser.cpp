// See onnx_tensor_parser.h. TensorProtoParser method bodies: TensorProto wire decode, external-data resolution, HostBuffer materialization.
#include "onnx_tensor_parser.h"

namespace vknn {
    namespace onnx {

        TensorProto TensorProtoParser::parse(Reader r) {
            TensorProto t;
            uint32_t    f, w;
            while (r.tag(f, w))
            {
                switch (f)
                {
                    case kTensorDims: // packed blob of varints, or a single varint
                        if (w == kWireBytes)
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
                    case kTensorDataType:
                        t.dataType = (int32_t) r.varint();
                        break;
                    case kTensorFloatData: // packed or single
                        if (w == kWireBytes)
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
                    case kTensorInt32Data: // packed or single: typed payload for INT32 and every
                                           // narrower type. A negative int32 arrives as a 64-bit
                                           // sign-extended varint; truncating the decoded u64 to
                                           // int32 recovers its value.
                        if (w == kWireBytes)
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
                    case kTensorInt64Data:
                        if (w == kWireBytes)
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
                    case kTensorDoubleData: // packed or single fixed64: real fp64, kept at full precision
                        if (w == kWireBytes)
                        {
                            Reader s = r.sub();
                            while (!s.eof())
                            {
                                uint64_t b = s.fixed64();
                                double   dv;
                                std::memcpy(&dv, &b, 8);
                                t.doubleData.push_back(dv);
                            }
                        } else
                        {
                            uint64_t b = r.fixed64();
                            double   dv;
                            std::memcpy(&dv, &b, 8);
                            t.doubleData.push_back(dv);
                        }
                        break;
                    case kTensorName:
                        t.name = r.str();
                        break;
                    case kTensorRawData:
                        t.raw = r.bytes();
                        break;
                    case kTensorExternalData: {
                        Reader      s = r.sub();
                        std::string key, val;
                        uint32_t    ef, ew;
                        while (s.tag(ef, ew))
                        {
                            if (ef == kStringEntryKey && ew == kWireBytes)
                            {
                                key = s.str();
                            } else if (ef == kStringEntryValue && ew == kWireBytes)
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
                    case kTensorDataLocation:
                        t.dataLocation = (int32_t) r.varint();
                        break;
                    default:
                        r.skip(w);
                        break;
                }
            }
            return t;
        }

        void TensorProtoParser::resolveExternal(const std::string &baseDir, TensorProto &t, std::map<std::string, std::vector<uint8_t>> &cache) {
            if (t.dataLocation != kDataLocationExternal || t.extLoc.empty() || !t.raw.empty())
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

        void TensorProtoParser::fillHostFloat(const TensorProto &t, HostBuffer &hb, int64_t elems) {
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

        void TensorProtoParser::fillHostBytes(const TensorProto &t, HostBuffer &hb, int64_t elems, DType dt) {
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

        void TensorProtoParser::fillHostDouble(const TensorProto &t, HostBuffer &hb, int64_t elems) {
            hb.resizeElems(elems, DType::Float64); // 8 bytes/elem, native fp64
            double *dst = hb.f64();
            if (!t.raw.empty() && isType(t.dataType, OnnxType::Double))
            {
                // raw_data DOUBLE: eight-byte lanes copied verbatim (clamp to the destination byte size,
                // see fillHostFloat: a negative elems sizes the buffer to 0 yet casts to a huge size_t).
                std::memcpy(dst, t.raw.data(), std::min<size_t>(t.raw.size(), hb.bytes.size()));
            } else if (!t.doubleData.empty())
            {
                for (int64_t i = 0; i < elems && i < (int64_t) t.doubleData.size(); ++i)
                {
                    dst[i] = t.doubleData[i];
                }
            }
        }

        void TensorProtoParser::fillHostI64(const TensorProto &t, HostBuffer &hb, int64_t elems) {
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

    } // namespace onnx
} // namespace vknn
