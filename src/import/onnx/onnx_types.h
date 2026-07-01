// ONNX data-type wire values (TensorProto.DataType) + the raw TensorProto holder used by the importer.
#pragma once
#include "vknn/graph.h"
#include <cstdint>
#include <string>
#include <vector>

namespace vknn {
    namespace onnx {

        // ONNX TensorProto.DataType wire values, used everywhere a tensor's element type is decoded so no
        // raw magic number (dtype "11", "10", ...) leaks into the parser.
        enum class OnnxType : int32_t {
            Undefined = 0,
            Float     = 1,
            Uint8     = 2,
            Int8      = 3,
            Uint16    = 4,
            Int16     = 5,
            Int32     = 6,
            Int64     = 7,
            String    = 8,
            Bool      = 9,
            Float16   = 10,
            Double    = 11,
            Uint32    = 12,
            Uint64    = 13,
        };
        inline constexpr bool isType(int32_t dt, OnnxType t) {
            return dt == (int32_t) t;
        }

        // ONNX -> vknn compute dtype. FLOAT / DOUBLE narrow to fp32; integers keep their width; anything
        // else is treated as fp32.
        inline DType dtypeFromElem(int32_t el) {
            switch ((OnnxType) el)
            {
                case OnnxType::Float16:
                    return DType::Float16;
                case OnnxType::Int64:
                    return DType::Int64;
                case OnnxType::Int32:
                    return DType::Int32;
                case OnnxType::Int8:
                    return DType::Int8;
                case OnnxType::Uint8:
                case OnnxType::Bool:
                    return DType::UInt8; // UINT8 / BOOL (0/1)
                default:
                    return DType::Float32; // FLOAT / DOUBLE (narrowed) / anything else -> fp32 compute
            }
        }

        // ----------------------------- TensorProto -----------------------------
        // fields: 1=dims(int64 repeated/packed), 2=data_type(int32), 4=float_data(packed),
        // 7=int64_data(packed), 8=name(string), 9=raw_data(bytes), 13=external_data
        // (repeated StringStringEntryProto{1=key,2=value}), 14=data_location(0=DEFAULT,1=EXTERNAL).
        // Large models (incl. anything torch's newer exporter emits) keep weights in a sibling .onnx.data
        // file and reference them via external_data; resolved against the model dir at materialize time.
        struct TensorProto {
            std::vector<int64_t> dims;
            int32_t              dataType = 1; // 1=FLOAT, 7=INT64
            std::string          name;
            std::vector<uint8_t> raw;
            std::vector<float>   floatData;
            std::vector<int64_t> int64Data;
            int32_t              dataLocation = 0; // 1 = EXTERNAL
            std::string          extLoc;           // external file (relative to the model dir)
            int64_t              extOffset = 0;
            int64_t              extLength = -1;
        };

    } // namespace onnx
} // namespace vknn
