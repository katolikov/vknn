// The graph IR. Backend-agnostic; every tensor here is NCHW.
#pragma once
#include "vknn/op.h"
#include "vknn/tensor.h"
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vknn {

    /// A whole model as a backend-agnostic intermediate representation: a flat tensor table, a node
    /// list, and the host data for constant initializers. Tensors are referenced everywhere by
    /// TensorId (an index into `tensors`), never by pointer, so the containers may reallocate freely.
    /// The importer and every graph pass mutate a Graph in place; a backend consumes it read-only.
    class Graph {
      public:
        /// Every tensor's shape/dtype descriptor, indexed by TensorId.
        std::vector<TensorDesc> tensors;
        /// The op instances. Topologically ordered after import (see topoSort()).
        std::vector<Node> nodes;
        /// Name -> id lookup for tensors that carry a name (graph I/O and ONNX-named intermediates).
        std::map<std::string, TensorId> tensorByName;
        /// Ids of the graph's external input tensors, in declaration order.
        std::vector<TensorId> inputs;
        /// Ids of the graph's external output tensors, in declaration order.
        std::vector<TensorId> outputs;
        /// Host-side constant data (weights, biases, shape constants) keyed by the tensor it backs.
        std::map<TensorId, HostBuffer> initializers;

        /// Look up `name`, appending a fresh tensor registered under `name` if none exists yet.
        /// @returns The id of the existing or newly created tensor.
        TensorId findOrAdd(const std::string &name);
        /// Look up an existing tensor by name without creating one.
        /// @returns Its id, or kNoTensor if no tensor is registered under `name`.
        TensorId find(const std::string &name) const;
        /// Append `d` to the tensor table.
        /// @returns The id (table index) of the newly added tensor.
        TensorId addTensor(TensorDesc d);
        /// True when `id` has constant host data in `initializers` (i.e. it is a weight/constant, not
        /// a runtime activation).
        bool isInitializer(TensorId id) const {
            return initializers.count(id) > 0;
        }

        /// The descriptor for `id`. Precondition: `id` is a valid index into `tensors`.
        const TensorDesc &desc(TensorId id) const {
            return tensors[id];
        }
        /// Mutable overload of desc(); same precondition.
        TensorDesc &desc(TensorId id) {
            return tensors[id];
        }

        /// Reorder `nodes` so every node follows its tensor producers (stable: nodes already in
        /// dependency order keep their relative positions). @throws Error if the graph contains a cycle.
        void topoSort();
        /// A readable, multi-line listing of the tensors and nodes, for debugging/logging.
        std::string dump() const;
    };

    /// Decode initializer `id`'s payload to fp32, honoring the stored dtype: a Float16 payload (an
    /// fp16 .vxm from vknn_compile) converts per element, a Float32 payload copies through. This is
    /// the one decode path for host-side payload reads — the session's CPU pool load and the Vulkan
    /// ops' weight prepacking both go through it, so a reader never reinterprets fp16 bytes as fp32.
    ///
    /// The element count comes from the tensor shape; a rank-0 scalar (shape [], numElements() 0)
    /// recovers its single element from the payload size instead, so the value is read, not dropped.
    /// @param g  Graph owning the initializer.
    /// @param id Initializer tensor id. Precondition: `g.isInitializer(id)`.
    /// @returns The payload as fp32 elements.
    inline std::vector<float> initFloats(const Graph &g, TensorId id) {
        const HostBuffer &hb = g.initializers.at(id);
        const DType       dt = g.desc(id).dtype;
        int64_t           n  = numElements(g.desc(id).shape);
        if (n <= 0)
        {
            const int64_t es = (int64_t) dtypeSize(dt); // native byte width of the stored payload (1 for int8/uint8)
            n                = es > 0 ? (int64_t) (hb.bytes.size() / es) : 0;
        }
        std::vector<float> out((size_t) std::max<int64_t>(n, 0));
        // Never decode more lanes than the stored blob actually holds: a .vxm / initializer whose
        // declared shape count exceeds its payload would otherwise read past hb.bytes (heap end, or a
        // mapped view's end -> SIGBUS). Each branch bounds its loop by the lanes its own read width
        // covers; any tail beyond the payload stays zero. A well-formed file has lanes(w) == n, so this
        // is a no-op there. (The else branch reads fp32 because FLOAT/DOUBLE/INT32 all materialize to
        // fp32 host bytes at import, so its width is 4 regardless of the declared dtype.)
        const size_t blobBytes = hb.bytes.size();
        auto         lanes     = [&](size_t w) -> int64_t {
            return w ? std::min<int64_t>(std::max<int64_t>(n, 0), (int64_t) (blobBytes / w)) : 0;
        };
        if (dt == DType::Float16)
        {
            const uint8_t *h = hb.bytes.data(); // viewed .vxm payload: unaligned, byte-copy only
            for (int64_t i = 0, lim = lanes(2); i < lim; ++i)
            {
                out[i] = halfToFloatAt(h, i);
            }
        } else if (dt == DType::Int64)
        {
            // Int64 initializers keep raw 8-byte lanes (never widened to fp32 at import), so a bare fp32
            // read would misinterpret the bytes. Const-folded boolean masks and shape/index constants reach
            // the flat GPU ops (And / Equal / Where cond) as Int64 -- decode the lanes to fp32 the same way
            // the CPU oracle reads them, so an Int64 constant operand loads its true 0/1 (or index) value.
            const int64_t *v = hb.i64();
            for (int64_t i = 0, lim = lanes(8); i < lim; ++i)
            {
                out[i] = (float) v[i];
            }
        } else if (dt == DType::UInt8)
        {
            // Native 1-byte quant lanes (a packed int4 / uint8 weight kept at its on-disk size by
            // materializeInitializers, never widened to fp32) decode to integer-valued fp32 -- the same
            // values the earlier fp32-widened storage held, so every reader is unaffected.
            const uint8_t *v = hb.bytes.data();
            for (int64_t i = 0, lim = lanes(1); i < lim; ++i)
            {
                out[i] = (float) v[i];
            }
        } else if (dt == DType::Int8)
        {
            const int8_t *v = reinterpret_cast<const int8_t *>(hb.bytes.data());
            for (int64_t i = 0, lim = lanes(1); i < lim; ++i)
            {
                out[i] = (float) v[i];
            }
        } else
        {
            // FLOAT / DOUBLE / INT32 materialize to fp32 host bytes at import, so a plain fp32 read recovers
            // the value.
            const float *f = hb.f32();
            for (int64_t i = 0, lim = lanes(4); i < lim; ++i)
            {
                out[i] = f[i];
            }
        }
        return out;
    }

    /// Import an ONNX model file into the backend-agnostic IR (canonical NCHW).
    Graph importOnnx(const std::string &path);

    /// Save the optimized graph (post-passes) as a compact self-contained ".vxm" binary, so a reload
    /// skips ONNX parsing + all graph passes. Writes the single-graph ("VXM3") container.
    /// @returns True on success; false if the file cannot be written.
    bool saveGraphBin(const Graph &g, const std::string &path);
    /// Load a ".vxm" binary into `g`, replacing its contents. Accepts both the single-graph ("VXM3")
    /// container and a multi-bucket ("VXM4") container -- for the latter the first bucket is taken, so
    /// single-graph callers keep working against either format.
    /// @returns True on success; false if the file is missing or not a valid ".vxm".
    bool loadGraphBin(Graph &g, const std::string &path);

    /// Save one or more shape buckets to a ".vxm". Each bucket is a full pass+plan graph for one
    /// declared input-shape set; buckets may differ in node identity but share ONE content-deduped
    /// initializer pool (identical weight payloads are stored once). A single bucket is written as the
    /// legacy single-graph ("VXM3") container so a fixed-shape model's bytes are unchanged; two or more
    /// buckets are written as the multi-bucket ("VXM4") container. `names` labels the buckets (a
    /// missing entry defaults to empty); its length need not match `buckets`.
    /// @returns True on success; false if `buckets` is empty or the file cannot be written.
    bool saveGraphBinBuckets(const std::vector<Graph> &buckets, const std::vector<std::string> &names, const std::string &path);
    /// Load every bucket from a ".vxm" into `buckets` (with per-bucket labels in `names`), replacing
    /// their contents. A legacy VXM3 file loads as exactly one bucket; a VXM4 file loads all of its
    /// buckets, each with the shared initializer payloads copied into its own initializer map.
    /// @returns True on success; false if the file is missing, truncated, or not a valid ".vxm".
    bool loadGraphBinBuckets(std::vector<Graph> &buckets, std::vector<std::string> &names, const std::string &path);
    /// Stream every bucket of a ".vxm" through `consume` one at a time, so only ONE bucket's weights
    /// are materialized in host memory at any moment — the difference between fitting and OOMing a
    /// device load whose buckets share multi-GB weights (the resident-pool + per-bucket-copy peak of
    /// loadGraphBinBuckets is the sum over all buckets). Shared pool payloads are read from the file
    /// region directly per bucket. `consume` receives the bucket graph (weights populated), its label,
    /// its index, and the bucket count; returning false stops the iteration early (not an error).
    /// @returns True when every consumed bucket read cleanly; false on a missing/truncated/foreign file.
    bool loadGraphBinBucketsStreamed(const std::string &path, const std::function<bool(Graph &&g, const std::string &name, size_t index, size_t count)> &consume);

} // namespace vknn
