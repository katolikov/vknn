// IR node struct: an op instance with its tensor ids, attributes, and fusion metadata.
#pragma once
#include "vknn/act_type.h"
#include "vknn/attributes.h"
#include "vknn/op_type.h"
#include "vknn/tensor_id.h"
#include <cstdint>
#include <string>
#include <vector>

namespace vknn {

    /// IR node. References tensors by id into Graph::tensors.
    struct Node {
        OpType                type = OpType::Unknown;
        std::string           name;
        std::vector<TensorId> inputs;
        std::vector<TensorId> outputs;
        Attributes            attr;
        // Fusion metadata filled by graph passes:
        ActType fusedAct = ActType::None;
        float   actLo = 0, actHi = 0;
        // For kUnary/kBinary: the UnaryType/BinaryType code. For unary ops with params (LeakyRelu/Elu
        // alpha, HardSigmoid alpha/beta) the params live in actLo/actHi.
        int32_t subOp = 0;
        // Conv only: a residual tensor fused into the epilogue (out = act(conv + residual)); set by the
        // residual-Add fusion pass. kNoTensor when not fused.
        TensorId fusedResidual = kNoTensor;
        // MatMul only: a rank-1 [N] bias initializer added into the fp32 accumulator before the store
        // (out[...,n] = matmul + bias[n]); set by the MatMul+bias fusion pass. kNoTensor when not fused.
        TensorId fusedBias = kNoTensor;
    };

} // namespace vknn
