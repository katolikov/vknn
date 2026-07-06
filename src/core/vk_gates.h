// The Vulkan backend's capability model as pure functions of the graph: which OpTypes have a GPU
// kernel and which node shapes/attributes that kernel accepts. Lives in core — compiled into every
// build, including hosts without the Vulkan backend — so `vknn_compile --support-report` and the
// host tests evaluate exactly the gate code the device engine runs, and the two can never drift.
#pragma once
#include "vknn/graph.h"
#include <string>
#include <vector>

namespace vknn {

    /// Whether a Vulkan kernel is registered for `t`. Mirrors the VKNN_REGISTER_VK_OP set in
    /// src/backend/vulkan/ops/; the device backend cross-checks this table against the live
    /// VkOpRegistry and warns on divergence (the registry itself only exists in Vulkan builds).
    bool vkKernelDeclared(OpType t);

    /// The Vulkan backend's shape/attribute gate: true when the GPU kernel for `nd` accepts this
    /// node's shapes, attributes, and operand constness. Pure function of the graph + node (no
    /// device state); VulkanBackend::supportsNode is this gate behind the availability/registry
    /// pre-checks. On refusal, fills `*whyNot` (when non-null) with a short stable
    /// "<Op>: <gate>" reason; a null `whyNot` costs nothing.
    bool vkNodeGate(const Graph &g, const Node &nd, std::string *whyNot = nullptr);

    /// One row of the support report: where a node lands and, when not on the GPU, why.
    struct NodeSupport {
        std::string node;    ///< Node name (diagnostics; may be empty).
        std::string op;      ///< ONNX-style op spelling (opTypeName).
        std::string backend; ///< "vulkan", "cpu", or "none" (no kernel in any backend).
        std::string reason;  ///< Refusal reason; empty when backend == "vulkan".
    };

    /// Per-node Vulkan-vs-CPU assignment of `g` under the capability model: a node runs on the GPU
    /// iff a kernel is declared and vkNodeGate accepts it, mirroring the session's plan-time
    /// assignment minus device availability, Config::disableVkOps, and the tiny-island fold. This
    /// is the code path behind `vknn_compile --support-report`.
    std::vector<NodeSupport> vkSupportSurvey(const Graph &g);

} // namespace vknn
