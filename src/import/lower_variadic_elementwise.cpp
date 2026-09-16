// Lower the variadic elementwise ONNX ops (Sum, Mean, Max, Min take 1..N operands) into the
// two-operand nodes every kernel implements. The importer maps Sum to OpType::Add, Max/Min to
// OpType::Binary with their BinaryType, and Mean to its own kernel-less OpType::Mean; the Binary/Add
// shape rule, constant folding, the Vulkan gate and both kernels read exactly inputs[0] and
// inputs[1], so a node with any other operand count must not survive import.
//
//   1 operand       -> Identity (the op of one value is that value; eliminateIdentity removes it).
//   N >= 2 operands -> a left fold of 2-input nodes of the same op: p1 = op(x0, x1),
//                      p2 = op(p1, x2), ..., the last writing the original output. NumPy
//                      broadcasting is associative over shapes, so the pairwise fold broadcasts to
//                      the same final shape.
//   Mean            -> the Add left fold into a sum tensor, then ONE Binary Mul by a rank-0 fp32
//                      initializer holding the fp32 reciprocal 1.0f / N: ONNX Runtime's CPU Mean sums
//                      left to right and scales the sum by that reciprocal, so the result is
//                      bit-identical to it (a Div by N rounds differently whenever N is not a power
//                      of two). The rank-0 scale broadcasts without changing the output rank.
//   0 operands      -> Error(InvalidArgument): the ONNX schema requires at least one.
#include "import/passes.h"
#include "vknn/binary_type.h"
#include "vknn/error.h"
#include "vknn/logging.h"
#include <string>
#include <vector>

namespace vknn {
    namespace {
        // Operand count of the Add/Binary kernels the variadic forms lower to.
        constexpr size_t kKernelOperandCount = 2;

        // True for a node this pass rewrites: every Mean, and a Sum (Add) or Max/Min (Binary) whose
        // operand count is not the kernels' two. Operands appended past pwCoreInputs by pointwise fusion
        // are not operands of the op itself.
        bool isVariadicElementwise(const Node &nd) {
            if (nd.type == OpType::Mean)
            {
                return true;
            }
            if (pwCoreInputs(nd) == kKernelOperandCount)
            {
                return false;
            }
            if (nd.type == OpType::Add)
            {
                return true;
            }
            const BinaryType binaryKind = (BinaryType) nd.subOp;
            return nd.type == OpType::Binary && (binaryKind == BinaryType::Max || binaryKind == BinaryType::Min);
        }

        // The ONNX spelling of the variadic op a node came from, for diagnostics.
        const char *variadicOnnxName(const Node &nd) {
            if (nd.type == OpType::Mean)
            {
                return "Mean";
            }
            if (nd.type == OpType::Add)
            {
                return "Sum";
            }
            return (BinaryType) nd.subOp == BinaryType::Max ? "Max" : "Min";
        }

        // `stem` when no tensor already carries that name, else `stem` with the smallest numeric
        // suffix that is free, so lowered intermediates never alias an existing tensor's name entry.
        std::string freeTensorName(const Graph &g, const std::string &stem) {
            if (g.find(stem) == kNoTensor)
            {
                return stem;
            }
            for (size_t suffix = 1;; ++suffix)
            {
                std::string candidate = stem + "_" + std::to_string(suffix);
                if (g.find(candidate) == kNoTensor)
                {
                    return candidate;
                }
            }
        }
    } // namespace

    void requireLoweredVariadicElementwise(const Graph &g) {
        for (const Node &nd: g.nodes)
        {
            if (isVariadicElementwise(nd))
            {
                throw Error(Status::InvalidArgument, std::string(variadicOnnxName(nd)) + " '" + nd.name + "': " + std::to_string(pwCoreInputs(nd)) + " operands; recompile the .vxm (variadic Sum/Mean/Max/Min are lowered to 2-input nodes at compile time)");
            }
        }
    }

    void lowerVariadicElementwise(Graph &g) {
        int               lowered = 0;
        std::vector<Node> appended;
        for (size_t i = 0; i < g.nodes.size(); ++i)
        {
            if (!isVariadicElementwise(g.nodes[i]))
            {
                continue;
            }
            // Copies, not references: g.nodes[i] is overwritten and addTensor may reallocate descs.
            const Node        original = g.nodes[i];
            const std::string onnxName = variadicOnnxName(original);
            const std::string label    = onnxName + " '" + original.name + "'";
            if (original.inputs.empty())
            {
                throw Error(Status::InvalidArgument, label + ": no inputs (the ONNX schema requires at least one operand)");
            }
            for (TensorId operand: original.inputs)
            {
                if (operand == kNoTensor)
                {
                    throw Error(Status::InvalidArgument, label + ": an operand input is missing (every operand is required)");
                }
            }
            if (original.outputs.empty() || original.outputs[0] == kNoTensor)
            {
                throw Error(Status::InvalidArgument, label + ": the result output is missing");
            }
            const TensorId output = original.outputs[0];
            const size_t   count  = original.inputs.size();
            ++lowered;

            if (count == 1)
            {
                Node identity;
                identity.type    = OpType::Identity;
                identity.name    = original.name;
                identity.inputs  = original.inputs;
                identity.outputs = {output};
                g.nodes[i]       = identity;
                continue;
            }

            const std::string stem       = original.name.empty() ? g.desc(output).name : original.name;
            const bool        isMean     = original.type == OpType::Mean;
            const DType       resultType = g.desc(output).dtype;
            // One accumulating step of the fold: the op of the running result and the next operand.
            auto foldStep = [&](size_t stepIndex, TensorId accumulated, TensorId operand, TensorId result) {
                Node step;
                step.type    = isMean ? OpType::Add : original.type;
                step.subOp   = isMean ? 0 : original.subOp;
                step.name    = stem + "#variadic_step" + std::to_string(stepIndex);
                step.inputs  = {accumulated, operand};
                step.outputs = {result};
                return step;
            };
            auto partialResult = [&](const std::string &suffix) {
                TensorDesc partial;
                partial.name  = freeTensorName(g, stem + suffix);
                partial.dtype = resultType;
                return g.addTensor(partial);
            };

            // Mean folds into a separate sum tensor that the final scale reads; the other ops' last
            // step writes the original output directly, so every consumer keeps its tensor id.
            const TensorId foldResult  = isMean ? partialResult("#mean_sum") : output;
            TensorId       accumulated = original.inputs[0];
            for (size_t k = 1; k < count; ++k)
            {
                const bool     lastStep = k + 1 == count;
                const TensorId result   = lastStep ? foldResult : partialResult("#variadic_partial" + std::to_string(k));
                const Node     step     = foldStep(k, accumulated, original.inputs[k], result);
                if (k == 1)
                {
                    g.nodes[i] = step; // in place; the rest are appended and topo-sorted below
                } else
                {
                    appended.push_back(step);
                }
                accumulated = result;
            }

            if (isMean)
            {
                TensorDesc reciprocalDesc;
                reciprocalDesc.name          = freeTensorName(g, stem + "#mean_reciprocal");
                reciprocalDesc.shape         = {}; // rank 0: broadcasts against the sum without changing its rank
                reciprocalDesc.dtype         = DType::Float32;
                reciprocalDesc.isInitializer = true;
                const TensorId reciprocalId  = g.addTensor(reciprocalDesc);
                HostBuffer     reciprocalValue;
                reciprocalValue.resizeElems(1, DType::Float32);
                reciprocalValue.f32()[0]     = 1.0f / (float) count; // computed in fp32, as ONNX Runtime does
                g.initializers[reciprocalId] = std::move(reciprocalValue);

                Node scale;
                scale.type    = OpType::Binary;
                scale.subOp   = (int32_t) BinaryType::Mul;
                scale.name    = stem + "#mean_scale";
                scale.inputs  = {foldResult, reciprocalId};
                scale.outputs = {output};
                appended.push_back(scale);
            }
        }
        if (!appended.empty())
        {
            for (Node &n: appended)
            {
                g.nodes.push_back(std::move(n));
            }
            g.topoSort();
        }
        if (lowered)
        {
            VKNN_INFO << "lowerVariadicElementwise: lowered " << lowered << " variadic Sum/Mean/Max/Min node(s) to 2-input form";
        }
    }

} // namespace vknn
