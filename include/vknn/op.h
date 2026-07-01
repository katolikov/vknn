// Op types, the fused-activation codes, the attribute bag, and the IR node struct.
#pragma once
#include "vknn/common.h"

// Umbrella header: the op-graph types now live one-per-file. Included in dependency order so a type
// used by another comes first. Every #include "vknn/op.h" keeps exposing the same names as before.
#include "vknn/tensor_id.h"  // TensorId, kNoTensor
#include "vknn/act_type.h"   // ActType
#include "vknn/op_type.h"    // OpType, opTypeName, opTypeFromOnnx
#include "vknn/unary_type.h" // UnaryType, unaryFromOnnx
#include "vknn/binary_type.h" // BinaryType, binaryFromOnnx
#include "vknn/reduce_type.h" // ReduceType, reduceFromOnnx
#include "vknn/attr.h"        // Attr
#include "vknn/attributes.h"  // Attributes
#include "vknn/node.h"        // Node
