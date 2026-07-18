// ONNX wire-format constants (per-message field numbers, TensorProto.DataType) + the raw
// TensorProto holder used by the importer. The protobuf wire types live with the Reader
// (onnx_reader.h); the framing is its contract, the message schema is this one.
#pragma once
#include "vknn/graph.h"
#include <cstdint>
#include <string>
#include <vector>

namespace vknn {
    namespace onnx {

        // Field numbers of the ONNX protobuf messages the importer consumes, matching onnx.proto.
        // Only consumed fields are named - unknown fields skip by wire type. Plain enums (not
        // enum class) so the values compare and switch directly against the decoded tag.
        enum ModelField : uint32_t {
            kModelGraph = 7, // ModelProto.graph (GraphProto)
        };
        enum GraphField : uint32_t {
            kGraphNode        = 1,  // GraphProto.node (repeated NodeProto)
            kGraphInitializer = 5,  // GraphProto.initializer (repeated TensorProto)
            kGraphInput       = 11, // GraphProto.input (repeated ValueInfoProto)
            kGraphOutput      = 12, // GraphProto.output
            kGraphValueInfo   = 13, // GraphProto.value_info (intermediate shape annotations)
        };
        enum NodeField : uint32_t {
            kNodeInput     = 1, // NodeProto.input (repeated string)
            kNodeOutput    = 2, // NodeProto.output
            kNodeName      = 3, // NodeProto.name
            kNodeOpType    = 4, // NodeProto.op_type
            kNodeAttribute = 5, // NodeProto.attribute (repeated AttributeProto)
        };
        enum AttrField : uint32_t {
            kAttrName   = 1, // AttributeProto.name
            kAttrFloat  = 2, // AttributeProto.f
            kAttrInt    = 3, // AttributeProto.i
            kAttrString = 4, // AttributeProto.s
            kAttrTensor = 5, // AttributeProto.t (TensorProto)
            kAttrFloats = 7, // AttributeProto.floats (packed repeated)
            kAttrInts   = 8, // AttributeProto.ints (packed repeated)
        };
        enum ValueInfoField : uint32_t {
            kValueInfoName = 1, // ValueInfoProto.name
            kValueInfoType = 2, // ValueInfoProto.type (TypeProto)
        };
        enum TypeField : uint32_t {
            kTypeTensorType = 1, // TypeProto.tensor_type
        };
        enum TensorTypeField : uint32_t {
            kTensorTypeElemType = 1, // TypeProto.Tensor.elem_type (TensorProto.DataType value)
            kTensorTypeShape    = 2, // TypeProto.Tensor.shape (TensorShapeProto)
        };
        enum ShapeField : uint32_t {
            kShapeDim = 1, // TensorShapeProto.dim (repeated Dimension)
        };
        enum DimField : uint32_t {
            kDimValue = 1, // Dimension.dim_value (concrete extent)
            kDimParam = 2, // Dimension.dim_param (symbolic axis name/expression)
        };
        enum TensorField : uint32_t {
            kTensorDims         = 1,  // TensorProto.dims (packed repeated int64)
            kTensorDataType     = 2,  // TensorProto.data_type (TensorProto.DataType value)
            kTensorFloatData    = 4,  // TensorProto.float_data (packed fixed32)
            kTensorInt32Data    = 5,  // TensorProto.int32_data (packed varints; also narrower types + fp16 bits)
            kTensorInt64Data    = 7,  // TensorProto.int64_data
            kTensorName         = 8,  // TensorProto.name
            kTensorRawData      = 9,  // TensorProto.raw_data (bytes)
            kTensorExternalData = 13, // TensorProto.external_data (repeated StringStringEntryProto)
            kTensorDataLocation = 14, // TensorProto.data_location (DataLocation value)
        };
        enum StringEntryField : uint32_t {
            kStringEntryKey   = 1, // StringStringEntryProto.key
            kStringEntryValue = 2, // StringStringEntryProto.value
        };
        // TensorProto.DataLocation values.
        enum DataLocation : int32_t {
            kDataLocationDefault  = 0, // payload inline in the model file
            kDataLocationExternal = 1, // payload in the external_data sibling file
        };

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
        // Compare a raw wire data_type field against a named OnnxType so call sites read as an intent
        // (isType(dt, OnnxType::Int64)) rather than a bare integer compare; the cast is on the enum,
        // keeping the parser's field value untouched.
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
        // Decoded TensorProto (field numbers above). Large models (incl. anything torch's newer
        // exporter emits) keep weights in a sibling .onnx.data file and reference them via
        // external_data; resolved against the model dir at materialize time.
        struct TensorProto {
            std::vector<int64_t> dims;
            int32_t              dataType = (int32_t) OnnxType::Float;
            std::string          name;
            std::vector<uint8_t> raw;
            std::vector<float>   floatData;
            std::vector<int32_t> int32Data; // typed payload for INT32 and narrower (INT8/UINT8/INT16/UINT16/BOOL) plus FLOAT16 bit patterns
            std::vector<int64_t> int64Data;
            int32_t              dataLocation = kDataLocationDefault;
            std::string          extLoc;           // external file (relative to the model dir)
            int64_t              extOffset = 0;
            int64_t              extLength = -1;
        };

    } // namespace onnx
} // namespace vknn
