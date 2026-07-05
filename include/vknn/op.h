// Op types, the fused-activation codes, the attribute bag, and the IR node struct.
#pragma once
#include "vknn/common.h"

// Umbrella header for the op-graph types (one type per included header). The includes are ordered by
// dependency so a type is defined before another that uses it.
#include "vknn/tensor_id.h"  // TensorId, kNoTensor
#include "vknn/act_type.h"   // ActType
#include "vknn/op_type.h"    // OpType, opTypeName, opTypeFromOnnx
#include "vknn/unary_type.h" // UnaryType, unaryFromOnnx
#include "vknn/binary_type.h" // BinaryType, binaryFromOnnx
#include "vknn/reduce_type.h" // ReduceType, reduceFromOnnx
#include "vknn/attr.h"        // Attr
#include "vknn/attributes.h"  // Attributes
#include "vknn/node.h"        // Node
