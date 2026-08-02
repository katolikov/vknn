#include "vk_segment.h"
#include "backend/cpu/parallel.h" // cpu::threadCount (host boundary pack/unpack partitioning)
#include "core/boundary_pack.h"   // parallel canonical<->boundary layout/precision conversion
#include "core/dispatch_tally.h"  // recorded-dispatch counter + per-node attribution
#include "core/kv_quant.h"        // int8 KV-cache scheme: eligibility rule + host codec (Hint::KvCacheQuant)
#include "core/matmul_tile.h"     // vec4-load routing + the activation row-pad rule
#include "core/matmul_view.h"     // kMmView (a view-addressed MatMul reads its own geometry, never a padded stride)
#include "core/quant_int4.h"      // kWq (a packed-quantized MatMul has its own operand layout)
#include "import/passes.h"        // readI64Param (raster-core view-eligibility diagnostic)
#include "ops/boundary_convert.h"
#include "ops/flat_ops.h" // flat::flatLocalSizeFor / laneWidthFor (family widths, resolved at load)
#include "vk_backend.h"
#include "vknn/dtype.h"
#include "vknn/logging.h"
#include "vknn/profiler.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

namespace vknn {

    namespace {
        // FNV-1a 64-bit hash constants. The model-tag hash keys the on-disk weight cache namespace and
        // the dma-buf id folds the same prime, so these values are part of the cache-key contract and
        // must not change.
        constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
        constexpr uint64_t kFnvPrime       = 1099511628211ull;
    } // namespace

    VulkanSegment::VulkanSegment(const std::vector<int> &idx, Graph &g, const Config &cfg, VulkanBackend *be): be_(be), g_(g), cfg_(cfg) {
        nodeIdx        = idx;
        useFp16_       = be_->useFp16(cfg);
        elemSize_      = useFp16_ ? 2 : 4;
        chainStepsMax_ = std::max(1, cfg.decodeChainSteps); // sizes the per-iteration link range sets + argmax result slots
        graphInputs_.insert(g.inputs.begin(), g.inputs.end());
        graphOutputs_.insert(g.outputs.begin(), g.outputs.end());

        // 1) allocate device buffers for all activation tensors (non-initializers).
        std::set<TensorId> acts;
        for (int ni: idx)
        {
            for (TensorId in: g.nodes[ni].inputs)
            {
                if (in != kNoTensor && !g.isInitializer(in))
                {
                    acts.insert(in);
                }
            }
            // A fused residual (out = act(conv + residual)) is read by record() but isn't in node.inputs,
            // so it needs a buffer too — and may be produced by another segment (boundary input).
            TensorId res = g.nodes[ni].fusedResidual;
            if (res != kNoTensor && !g.isInitializer(res))
            {
                acts.insert(res);
            }
            for (TensorId o: g.nodes[ni].outputs)
            {
                if (o != kNoTensor)
                {
                    acts.insert(o);
                }
            }
        }
        // Tensors this segment produces that are read OUTSIDE it (by another segment or as a graph
        // output) get downloaded to host via unpackFromBuffer. The default kAuto memory is
        // write-combined (fast to upload, but CPU READS are uncached and brutally slow -> 150ms on a
        // YOLO head boundary). Allocate those readback buffers as HOST_CACHED so the download is fast;
        // keep the rest as kAuto.
        std::set<int>      idxSet(idx.begin(), idx.end());
        std::set<TensorId> readBack(g.outputs.begin(), g.outputs.end());
        {
            std::set<TensorId> produced;
            for (int ni: idx)
            {
                for (TensorId o: g.nodes[ni].outputs)
                {
                    if (o != kNoTensor)
                    {
                        produced.insert(o);
                    }
                }
            }
            for (size_t q = 0; q < g.nodes.size(); ++q)
            {
                if (idxSet.count((int) q))
                {
                    continue;
                }
                for (TensorId in: g.nodes[q].inputs)
                {
                    if (in != kNoTensor && produced.count(in))
                    {
                        readBack.insert(in);
                    }
                }
            }
        }
        // Debug: Config::dumpTensors="substr1,substr2" forces matching tensors to dedicated (un-aliased)
        // readback buffers and dumps them to cfg.layerDumpDir after the run — so intermediate
        // activations can be diffed despite the liveness planner reusing buffers. A few tensors only.
        if (!cfg_.dumpTensors.empty())
        {
            std::string list = cfg_.dumpTensors;
            for (TensorId tid: acts)
            {
                const std::string &nm = g.tensors[tid].name;
                if (nm.empty())
                {
                    continue;
                }
                size_t pos = 0, comma;
                do
                {
                    comma           = list.find(',', pos);
                    std::string sub = list.substr(pos, comma == std::string::npos ? comma : comma - pos);
                    if (!sub.empty() && nm.find(sub) != std::string::npos)
                    {
                        readBack.insert(tid);
                        dumpTids_.push_back(tid);
                        break;
                    }
                    pos = comma + 1;
                } while (comma != std::string::npos);
            }
        }
        // int8 KV cache (Hint::KvCacheQuant): the eligible cache tensors — the ONE shared rule in
        // core/kv_quant.h, gated here on the device's 8-bit storage support and the fp16 session —
        // store a 1-byte payload in buffers_ plus an fp16 per-(head, token)-row scale side buffer.
        // With the hint Off (the default) the set is empty and every allocation below is
        // byte-identical to the fp16 path.
        {
            const auto &deviceCaps  = be_->ctx().caps();
            const bool  int8Storage = deviceCaps.storage8bit && deviceCaps.shaderInt8;
            for (TensorId tid: kvQuantCacheTensors(g, cfg, useFp16_ && int8Storage, /*requireFlat=*/true))
            {
                for (int ni: idx)
                {
                    const Node &node = g.nodes[ni];
                    if (!kvQuantNodeEligible(g, node) || (node.inputs[1] != tid && node.inputs[2] != tid))
                    {
                        continue;
                    }
                    KvqCache cache;
                    cache.headDim   = node.attr.geti(kFaHd);
                    cache.rows      = numElements(g.tensors[tid].shape) / cache.headDim;
                    kvqCaches_[tid] = cache; // the scale buffer is allocated with the boundary buffers below
                    break;
                }
            }
            if (!kvqCaches_.empty())
            {
                VKNN_INFO << "int8 KV cache: " << kvqCaches_.size() << " cache tensor(s) stored as int8 payload + fp16 row scales";
            } else if (cfg.kvCacheQuantMode() == (int) Mode::On)
            {
                // Named refusal: the hint asked for the scheme but nothing qualified on this
                // segment (no eligible split-KV attention, an fp32 session, or a device without
                // 8-bit storage). The fp16 cache path runs unchanged.
                VKNN_INFO << "int8 KV cache requested but no eligible cache tensor on this segment: " << kvQuantGraphRefusal(g) << " (device int8 storage: " << (int8Storage ? "yes" : "no") << ", fp16 storage: " << (useFp16_ ? "yes" : "no") << ")";
            }
        }
        // Virtualized activation row stride: give an internal flat fp16 activation a buffer whose
        // physical last axis is roundUpVec4(last) when that is exactly what unlocks the vec4-load
        // GEMM route on every one of its consumers (core/matmul_tile.h). The attention batched
        // MatMuls are the case that pays: a token count indivisible by 4 makes every row of the
        // softmax probabilities (the A operand, last axis == K) and of the transposed keys (the B
        // operand, last axis == N) start at an unaligned element index, which no partial-tail trick
        // can rescue — only the physical stride can. Both sides of the contract come from the same
        // rule, so the layout and the kernel that reads it are decided once.
        // A padded store writes row o at o * padded rather than o * last, so it must never land in
        // the buffer its own kernel is still reading: the liveness pool below frees a slot only once
        // its occupant's last use is STRICTLY before the new tensor's first use, and a producer's
        // input dies at the producing node itself — so a node's input and output never share a slot.
        {
            std::map<TensorId, int>              producerIn;  // tensor -> producing node index inside this segment
            std::map<TensorId, std::vector<int>> consumersOf; // tensor -> every consuming node index, whole graph
            for (int ni: idx)
            {
                for (TensorId o: g.nodes[ni].outputs)
                {
                    if (o != kNoTensor)
                    {
                        producerIn[o] = ni;
                    }
                }
            }
            for (size_t q = 0; q < g.nodes.size(); ++q)
            {
                for (TensorId in: g.nodes[q].inputs)
                {
                    if (in != kNoTensor)
                    {
                        consumersOf[in].push_back((int) q);
                    }
                }
                // A fused residual/bias is read by record() without appearing in node.inputs, so it
                // counts as a consumer here too — a reader that indexes the logical layout must
                // veto the padding just like any other.
                for (TensorId fused: {g.nodes[q].fusedResidual, g.nodes[q].fusedBias})
                {
                    if (fused != kNoTensor)
                    {
                        consumersOf[fused].push_back((int) q);
                    }
                }
            }
            // A producer qualifies when its kernel can store at a padded last-axis stride: the flat
            // gather (a Transpose that actually permutes; an identity one is aliased onto its input
            // by the pure-copy rule below and never runs) and the flat last-axis Softmax.
            auto producerStoresPadded = [&](const Node &nd, TensorId tid) {
                if (nd.outputs.empty() || nd.outputs[0] != tid)
                {
                    return false; // only the primary store moves; a pw chain's extra output streams
                                  // index the flat logical world and would miss the padded rows
                }
                if (nd.type == OpType::Transpose)
                {
                    if (nd.attr.has("view_stride"))
                    {
                        return true; // a folded movement chain always runs the gather
                    }
                    const auto &perm = nd.attr.getints("perm");
                    if (perm.empty())
                    {
                        return false;
                    }
                    for (size_t d = 0; d < perm.size(); ++d)
                    {
                        if (perm[d] != (int64_t) d)
                        {
                            return true;
                        }
                    }
                    return false; // identity perm: a pure copy, aliased rather than dispatched
                }
                if (nd.type != OpType::Softmax)
                {
                    return false;
                }
                const Shape &s    = g.desc(nd.inputs[0]).shape;
                int64_t      axis = nd.attr.geti("axis", -1);
                if (axis < 0)
                {
                    axis += (int64_t) s.size();
                }
                return axis == (int64_t) s.size() - 1; // padding moves the LAST axis; only a last-axis softmax fits
            };
            for (const auto &entry: producerIn)
            {
                const TensorId tid = entry.first;
                const auto    &td  = g.tensors[tid];
                if (!useFp16_ || !td.gpuFlat || td.storeFp32 || td.isInitializer || td.shape.size() < 2 || kvqCaches_.count(tid) || readBack.count(tid) ||
                    graphInputs_.count(tid))
                {
                    continue; // only an internal, flat, fp16 activation may change physical layout
                }
                if (!producerStoresPadded(g.nodes[entry.second], tid))
                {
                    continue;
                }
                auto consumers = consumersOf.find(tid);
                if (consumers == consumersOf.end() || consumers->second.empty())
                {
                    continue;
                }
                int64_t padded = 0;
                bool    ok     = true;
                for (int ci: consumers->second)
                {
                    const Node &use = g.nodes[ci];
                    // Every consumer must be a dense tiled MatMul in THIS segment reading the tensor
                    // through the padded stride: a view-addressed or packed-quantized MatMul, a
                    // fused-bias/residual read, or any other op would read the logical layout.
                    if (!idxSet.count(ci) || use.type != OpType::MatMul || use.attr.has(kMmView) || use.attr.has(kWq) || use.inputs.size() < 2 || use.inputs[0] == use.inputs[1] || nodeFp32(use) || use.fusedResidual == tid || use.fusedBias == tid)
                    {
                        ok = false;
                        break;
                    }
                    const bool isA = use.inputs[0] == tid;
                    const bool isB = use.inputs[1] == tid;
                    if (!isA && !isB)
                    {
                        ok = false;
                        break;
                    }
                    int64_t want = 0;
                    if (!matmulVec4PadUnlocks(useFp16_, isA ? MatMulOperand::A : MatMulOperand::B, g.desc(use.inputs[0]).shape, g.desc(use.inputs[1]).shape,
                                              g.desc(use.outputs[0]).shape, want) ||
                        (padded != 0 && want != padded))
                    {
                        ok = false;
                        break;
                    }
                    padded = want;
                }
                if (ok && padded > td.shape.back())
                {
                    rowPad_[tid] = padded;
                }
            }
            if (!rowPad_.empty())
            {
                VKNN_INFO << "vec4 activation padding: " << rowPad_.size() << " tensor(s) allocated with a 4-aligned physical row stride to unlock the vec4-load GEMM";
            }
        }
        auto actBytes = [&](TensorId tid) -> size_t {
            int64_t elems = g.tensors[tid].gpuFlat ? numElements(g.tensors[tid].shape) : packedElems(g.tensors[tid].shape);
            auto    padIt = rowPad_.find(tid);
            if (padIt != rowPad_.end())
            {
                // Physical extent: the same tensor with its last axis widened to the padded stride.
                elems = elems / g.tensors[tid].shape.back() * padIt->second;
            }
            if (kvqCaches_.count(tid))
            {
                // int8 KV cache payload: 1 byte per element (the fp16 row scales live in the
                // dedicated side buffer, never in this activation buffer).
                return (size_t) elems;
            }
            int    es = g.tensors[tid].storeFp32 ? 4 : elemSize_; // selective-fp32 tensors keep 4-byte storage
            size_t b  = (size_t) elems * es;
            return b == 0 ? (size_t) elemSize_ * 4 : b;
        };
        // Liveness buffer planner. One buffer per tensor keeps ALL activations live at once (~11.5GB on
        // the YoNoSplat encoder); the simultaneously-live peak is ~0.17GB. Boundary-in (produced by
        // another segment) and readback (read by host / another segment) tensors get dedicated buffers;
        // internal (produced-and-consumed only here) tensors are pooled by a greedy scan over execution
        // order, reusing a buffer once its previous occupant's last use has passed. The buffer-level
        // barriers in record() make the write-after-read at each reuse point safe.
        std::set<TensorId> producedHere;
        for (int ni: idx)
        {
            for (TensorId o: g.nodes[ni].outputs)
            {
                if (o != kNoTensor)
                {
                    producedHere.insert(o);
                }
            }
        }
        // Geometry-as-metadata: a pure-copy op copies its input verbatim, so alias the output onto the
        // input's buffer and skip the copy dispatch (record() checks src==dst). A Reshape/Squeeze/Unsqueeze
        // qualifies whenever input and output share layout + byte size (the layout pass guarantees this,
        // else it inserts a convert). A full-range unit-step Slice (start=0, step=1 on every axis) is the
        // same thing — an x[:] no-op that survives import; the byte-size guard below proves same shape, so
        // start=0 + step=1 makes it a verbatim copy. Restricted to an internal (poolable) output so
        // readback/boundary tensors keep their own buffer; the root input's liveness is extended to the
        // output's consumers because the liveness scan resolves aliases below.
        std::map<TensorId, TensorId> aliasRoot;
        auto                         resolveAlias = [&](TensorId t) {
            for (int hop = 0; hop < 64 && aliasRoot.count(t); ++hop)
            {
                t = aliasRoot[t];
            }
            return t;
        };
        // A Transpose whose perm is the identity permutes nothing: byte-for-byte copy. An ABSENT
        // perm means full axis reversal (the ONNX default), so only an explicit iota perm (or a
        // rank<=1 tensor, where reversal IS the identity) qualifies.
        auto isIdentityTranspose = [&](const Node &nd) {
            // A folded movement chain (view_stride) overrides the perm; its map is never identity-checkable here.
            if (nd.type != OpType::Transpose || nd.inputs.empty() || nd.inputs[0] == kNoTensor || nd.attr.has("view_stride"))
            {
                return false;
            }
            const auto &perm = nd.attr.getints("perm");
            if (perm.empty())
            {
                return g.desc(nd.inputs[0]).shape.size() <= 1;
            }
            for (size_t d = 0; d < perm.size(); ++d)
            {
                if (perm[d] != (int64_t) d)
                {
                    return false;
                }
            }
            return true;
        };
        auto isIdentitySlice = [&](const Node &nd) {
            if (nd.type != OpType::Slice || nd.attr.has("view_stride"))
            {
                return false; // a folded chain's map overrides starts/steps
            }
            auto starts = readI64Param(g, nd, "starts", 1);
            if (starts.empty())
            {
                return false; // params must be visible (static) to prove the slice is a no-op
            }
            for (int64_t s: starts)
            {
                if (s != 0)
                {
                    return false;
                }
            }
            for (int64_t s: readI64Param(g, nd, "steps", 4))
            {
                if (s != 1)
                {
                    return false; // a non-unit or negative step reorders elements, not a copy
                }
            }
            return true;
        };
        for (int ni: idx)
        {
            const Node &nd = g.nodes[ni];
            // A geometry op carrying a fused pointwise epilogue (pw_steps) is NOT a pure copy — its
            // kernel must run to apply the chain, so it can't be aliased/skipped.
            bool pureCopy = !nd.attr.has("pw_steps") && (nd.type == OpType::Reshape || nd.type == OpType::Flatten || nd.type == OpType::Squeeze || nd.type == OpType::Unsqueeze || isIdentitySlice(nd) || isIdentityTranspose(nd));
            if (!pureCopy || nd.inputs.empty() || nd.outputs.empty())
            {
                continue;
            }
            TensorId in = nd.inputs[0], out = nd.outputs[0];
            if (in == kNoTensor || out == kNoTensor || g.isInitializer(in))
            {
                continue;
            }
            if (!producedHere.count(out) || readBack.count(out) || g.tensors[out].storeFp32 || g.tensors[in].storeFp32)
            {
                continue; // output must be internal; skip fp32-pinned tensors
            }
            if (actBytes(in) != actBytes(out) || g.tensors[in].gpuFlat != g.tensors[out].gpuFlat)
            {
                continue; // only a byte-for-byte, same-layout reshape can share the buffer
            }
            aliasRoot[out] = resolveAlias(in);
        }
        // Zero-copy Concat/Split: when a Concat's parts (or a Split's outputs) are contiguous slices
        // of the whole in the STORED byte layout, each slice gets a sub-buffer VIEW into the whole's
        // memory instead of its own buffer. Producers then write their slice of the concatenation in
        // place and split consumers read theirs in place, so the Concat/Split node records nothing —
        // record() skips a slice exactly when the buffer identity proves the planner created its view.
        // Bit-exact by construction: the bytes and the arithmetic producing them are unchanged, only
        // the copies disappear. Links compose: a Split of a Concat output views straight into the
        // concat arena, and a concat-of-all-features chain (DenseNet) collapses to ONE arena per dense
        // block because each smaller concat output re-links as a prefix view of the next. Contiguity:
        // NC4HW4 tiles per channel block (rank 4, N==1, axis 1, every slice C%4==0); flat tiles
        // whenever every dim before the axis is 1. Any ineligible node keeps the dispatching path.
        std::map<TensorId, std::pair<TensorId, size_t>> viewOf; // member -> (parent, byte offset within parent)
        auto                                            resolveView = [&](TensorId t) {
            // A chain deeper than the cap resolves to kNoTensor rather than a mid-chain tensor: a
            // truncated resolution would link against a non-root and silently re-parent its subtree.
            // Every caller treats kNoTensor as "ineligible", so an absurdly deep chain only loses
            // the optimization.
            // Deeper than any real chain (a DenseNet-264 block links ~48 generations); named so the
            // bound is visibly a guard, not a tuning value.
            constexpr int kViewChainMaxHops = 4096;
            size_t        off               = 0;
            for (int hop = 0; hop < kViewChainMaxHops; ++hop)
            {
                auto it = viewOf.find(t);
                if (it == viewOf.end())
                {
                    return std::make_pair(t, off);
                }
                off += it->second.second;
                t = it->second.first;
            }
            return std::make_pair(kNoTensor, size_t(0));
        };
        // WRITE intervals committed into each arena: every concat part's producer writes its slice,
        // and a re-rooted group's producers write the group's whole extent, so two links whose
        // intervals overlap would let two writers race for the same bytes (schedule-dependent
        // corruption). A link may only claim a range disjoint from every other claim on that arena;
        // split/slice views claim nothing (they are read-only aliases of bytes the parent's own
        // writers produce).
        std::map<TensorId, std::vector<std::pair<size_t, size_t>>> writeClaims;
        auto                                                       claimWrite = [&](TensorId parent, size_t begin, size_t end) {
            auto &intervals = writeClaims[parent];
            for (const auto &c: intervals)
            {
                if (begin < c.second && c.first < end)
                {
                    return false;
                }
            }
            intervals.push_back({begin, end});
            return true;
        };
        int viewSlices = 0, viewSites = 0;
        {
            // A tensor may hold a view only when its buffer is byte-for-byte the slice: same storage
            // width (no fp32 pins), same layout world, produced by a GPU node of THIS segment, and not
            // host-read (readback tensors keep their dedicated HOST_CACHED buffers).
            auto memberOk = [&](TensorId t, bool wantFlat) {
                return t != kNoTensor && !g.isInitializer(t) && producedHere.count(t) && !readBack.count(t) && !g.tensors[t].storeFp32 && g.tensors[t].gpuFlat == wantFlat;
            };
            // Execution-order positions for the liveness rescue below: the last position reading a
            // tensor and the position producing it. Positions index into `idx` (the segment's
            // execution order), so "reader < writer" compares real recording order.
            std::map<TensorId, int> lastReadPos, producerPos;
            for (int p = 0; p < (int) idx.size(); ++p)
            {
                const Node &pn = g.nodes[idx[p]];
                for (TensorId in: pn.inputs)
                {
                    if (in != kNoTensor)
                    {
                        lastReadPos[in] = p;
                    }
                }
                if (pn.fusedResidual != kNoTensor)
                {
                    lastReadPos[pn.fusedResidual] = p;
                }
                for (TensorId o: pn.outputs)
                {
                    if (o != kNoTensor)
                    {
                        producerPos[o] = p;
                    }
                }
            }
            for (int ni: idx)
            {
                const Node &nd = g.nodes[ni];
                // Leading-axis constant Pad: pad(x) along one axis with every dim before it equal
                // to 1 is structurally concat(fill, x, fill) — x is a contiguous sub-range of the
                // output at the begin-pad offset. x's producer then writes through a view into the
                // padded buffer and the Pad records only vkCmdFillBuffer for the pad ranges (a
                // transfer-stage write; transferFillNodes_ routes the barrier kind). flat::Pad
                // mirrors this rule and takes the fill path only on proven view identity.
                if (nd.type == OpType::Pad)
                {
                    if (nd.attr.has("pw_steps") || nd.fusedResidual != kNoTensor || nd.inputs.empty() || nd.outputs.size() != 1)
                    {
                        continue;
                    }
                    TensorId data = nd.inputs[0], out = nd.outputs[0];
                    if (data == kNoTensor || out == kNoTensor || g.isInitializer(data) || g.tensors[out].storeFp32 || !g.tensors[out].gpuFlat || !memberOk(data, true))
                    {
                        continue;
                    }
                    if (nd.attr.gets("mode", "constant") != std::string("constant"))
                    {
                        continue;
                    }
                    // A runtime pad VALUE cannot fill at record time (its bytes are not known when
                    // the command buffer is recorded); an initializer or attribute value can.
                    if (nd.inputs.size() > 2 && nd.inputs[2] != kNoTensor && !g.isInitializer(nd.inputs[2]))
                    {
                        continue;
                    }
                    const Shape &is = g.desc(data).shape, &os = g.desc(out).shape;
                    const int    rank = (int) os.size();
                    if ((int) is.size() != rank || rank == 0 || numElements(is) <= 0 || numElements(os) <= 0)
                    {
                        continue;
                    }
                    auto pads = readI64Param(g, nd, "pads", 1);
                    if ((int) pads.size() < 2 * rank)
                    {
                        continue;
                    }
                    int  padAxis = -1;
                    bool padOk   = true;
                    for (int d = 0; d < rank && padOk; ++d)
                    {
                        const int64_t b = pads[(size_t) d], e = pads[(size_t) (rank + d)];
                        if (b < 0 || e < 0)
                        {
                            padOk = false; // negative pads crop, not pad
                        } else if (b > 0 || e > 0)
                        {
                            padOk   = padAxis < 0;
                            padAxis = d;
                        }
                    }
                    if (!padOk || padAxis < 0)
                    {
                        continue; // multi-axis pads interleave; a no-pad Pad is an identity handled elsewhere
                    }
                    int64_t outer = 1;
                    for (int d = 0; d < padAxis; ++d)
                    {
                        outer *= os[(size_t) d];
                    }
                    if (outer != 1)
                    {
                        continue;
                    }
                    size_t stride = 1;
                    for (int d = padAxis + 1; d < rank; ++d)
                    {
                        stride *= (size_t) os[(size_t) d];
                    }
                    const size_t offBytes  = (size_t) pads[(size_t) padAxis] * stride * elemSize_;
                    const size_t dataBytes = (size_t) numElements(is) * elemSize_;
                    const size_t outBytes  = (size_t) numElements(os) * elemSize_;
                    // vkCmdFillBuffer needs 4-byte-aligned offsets and sizes for both pad ranges.
                    if (offBytes % sizeof(uint32_t) != 0 || (offBytes + dataBytes) % sizeof(uint32_t) != 0 || outBytes % sizeof(uint32_t) != 0)
                    {
                        continue;
                    }
                    // The data tensor must be its own group root (a bigger group would drag other
                    // live bytes under the fill ranges); the pad output claims every byte — the
                    // producer writes the data range, the fills write the rest.
                    const TensorId dataRoot = resolveAlias(data);
                    const auto     ro       = resolveView(dataRoot);
                    if (ro.first != dataRoot || ro.second != 0 || actBytes(dataRoot) != dataBytes || dataRoot == out || offBytes + dataBytes > actBytes(out))
                    {
                        continue;
                    }
                    if (!claimWrite(out, 0, outBytes))
                    {
                        continue;
                    }
                    viewOf[dataRoot] = {out, offBytes};
                    ++viewSlices;
                    ++viewSites;
                    transferFillNodes_.insert(ni);
                    continue;
                }
                // Contiguous unit-step Slice: when the sliced box is a flat sub-range of the input
                // (leading dims select one index, at most one axis is partial, trailing dims are
                // full), the output is the input's bytes at a fixed offset — a sub-buffer view, and
                // the gather dispatch disappears. SliceOp::record re-derives the same contiguity rule
                // and skips only on a proven view identity.
                if (nd.type == OpType::Slice)
                {
                    if (nd.attr.has("pw_steps") || nd.attr.has("view_stride") || nd.fusedResidual != kNoTensor || nd.inputs.empty() || nd.outputs.size() != 1)
                    {
                        continue; // view_stride: the composed map overrides starts/steps
                    }
                    TensorId in0 = nd.inputs[0], out = nd.outputs[0];
                    if (in0 == kNoTensor || out == kNoTensor || g.isInitializer(in0) || g.tensors[in0].storeFp32 || !g.tensors[in0].gpuFlat || !memberOk(out, true))
                    {
                        continue;
                    }
                    const Shape &is = g.desc(in0).shape, &os = g.desc(out).shape;
                    const int    r = (int) is.size();
                    if ((int) os.size() != r || r == 0 || numElements(os) <= 0 || numElements(is) <= 0)
                    {
                        continue;
                    }
                    auto startsP = readI64Param(g, nd, "starts", 1);
                    auto axesP   = readI64Param(g, nd, "axes", 3);
                    auto stepsP  = readI64Param(g, nd, "steps", 4);
                    if (startsP.empty())
                    {
                        continue; // params must be static to prove the box
                    }
                    std::vector<int64_t> start(r, 0), step(r, 1);
                    for (size_t a = 0; a < startsP.size(); ++a)
                    {
                        int ax = axesP.empty() ? (int) a : (int) (axesP[a] < 0 ? axesP[a] + r : axesP[a]);
                        if (ax < 0 || ax >= r)
                        {
                            continue;
                        }
                        int64_t s0 = startsP[a] < 0 ? startsP[a] + is[ax] : startsP[a];
                        start[ax]  = std::max<int64_t>(0, std::min<int64_t>(s0, is[ax]));
                        step[ax]   = a < stepsP.size() ? stepsP[a] : 1;
                    }
                    bool boxOk   = true;
                    int  partial = -1; // the last dim whose range is not the full axis
                    for (int d = 0; d < r; ++d)
                    {
                        if (step[d] != 1 || start[d] + os[d] > is[d])
                        {
                            boxOk = false;
                            break;
                        }
                        if (start[d] != 0 || os[d] != is[d])
                        {
                            partial = d;
                        }
                    }
                    for (int d = 0; boxOk && partial >= 0 && d < r; ++d)
                    {
                        if ((d < partial && os[d] != 1) || (d > partial && (start[d] != 0 || os[d] != is[d])))
                        {
                            boxOk = false;
                        }
                    }
                    if (!boxOk || partial < 0 || viewOf.count(out))
                    {
                        continue; // partial < 0 = identity slice, already aliased whole-buffer
                    }
                    size_t offElems = 0;
                    {
                        std::vector<int64_t> strides(r, 1);
                        for (int d = r - 2; d >= 0; --d)
                        {
                            strides[d] = strides[d + 1] * is[d + 1];
                        }
                        for (int d = 0; d < r; ++d)
                        {
                            offElems += (size_t) start[d] * (size_t) strides[d];
                        }
                    }
                    const TensorId parent = resolveAlias(in0);
                    // Same session-input refusal as Split: a dma-buf rebind of a session input
                    // would orphan views into the replaced allocation.
                    if (resolveView(parent).first == out || graphInputs_.count(parent))
                    {
                        continue;
                    }
                    const size_t offBytes = offElems * elemSize_;
                    if (offBytes + actBytes(out) > actBytes(parent))
                    {
                        continue;
                    }
                    viewOf[out] = {parent, offBytes};
                    ++viewSlices;
                    ++viewSites;
                    fullyElided_.insert(ni); // the lone output aliased: record() emits nothing
                    continue;
                }
                const bool isConcat = nd.type == OpType::Concat;
                if ((!isConcat && nd.type != OpType::Split) || nd.attr.has("pw_steps") || nd.fusedResidual != kNoTensor)
                {
                    continue;
                }
                // whole = the concatenated output / the split input; slices = the parts / the outputs.
                if (isConcat ? (nd.outputs.size() != 1 || nd.inputs.empty()) : (nd.inputs.empty() || nd.outputs.empty()))
                {
                    continue;
                }
                TensorId whole = isConcat ? nd.outputs[0] : nd.inputs[0];
                if (whole == kNoTensor || g.isInitializer(whole) || g.tensors[whole].storeFp32)
                {
                    continue;
                }
                const std::vector<TensorId> &slices = isConcat ? nd.inputs : nd.outputs;
                const Shape                 &ws     = g.desc(whole).shape;
                const bool                   flat   = g.tensors[whole].gpuFlat;
                const int                    rank   = (int) ws.size();
                int64_t                      axis   = nd.attr.geti("axis", isConcat ? 1 : 0);
                if (axis < 0)
                {
                    axis += rank;
                }
                if (axis < 0 || axis >= rank || numElements(ws) <= 0)
                {
                    continue;
                }
                if (flat)
                {
                    int64_t outer = 1;
                    for (int d = 0; d < (int) axis; ++d)
                    {
                        outer *= ws[d];
                    }
                    if (outer != 1)
                    {
                        continue; // slices interleave along an inner axis: not contiguous slabs
                    }
                } else if (rank != 4 || ws[0] != 1 || axis != 1)
                { continue; }
                // Byte offset and size of each slice within the whole, refusing any structural
                // mismatch. All members store elemSize_ bytes per element (fp32 pins are refused
                // above/below), so flat offsets count elements and NC4HW4 offsets count whole
                // channel blocks.
                std::vector<size_t> offs(slices.size()), sliceBytes(slices.size());
                bool                shapeOk = true;
                int64_t             axisSum = 0;
                size_t              run     = 0;
                for (size_t si = 0; si < slices.size() && shapeOk; ++si)
                {
                    TensorId t = slices[si];
                    if (t == kNoTensor)
                    {
                        shapeOk = false;
                        break;
                    }
                    const Shape &ss = g.desc(t).shape;
                    shapeOk         = (int) ss.size() == rank && numElements(ss) > 0;
                    for (int d = 0; shapeOk && d < rank; ++d)
                    {
                        shapeOk = d == (int) axis ? ss[d] >= 1 : ss[d] == ws[d];
                    }
                    if (!shapeOk)
                    {
                        break;
                    }
                    if (!flat && ss[1] % kNC4Block != 0)
                    {
                        shapeOk = false; // an unaligned slice straddles a channel block
                        break;
                    }
                    offs[si]       = run;
                    sliceBytes[si] = (size_t) (flat ? numElements(ss) : packedElems(ss)) * elemSize_;
                    run += sliceBytes[si];
                    axisSum += ss[axis];
                }
                if (!shapeOk || axisSum != ws[axis])
                {
                    continue;
                }
                const TensorId wholeRoot = resolveView(resolveAlias(whole)).first;
                size_t         linked    = 0;
                if (isConcat)
                {
                    // A slice's whole GROUP re-roots into the concat arena, so a root r may link
                    // ONLY when this concat's slices with root r exactly TILE r's bytes with one
                    // uniform delta — then every byte of the re-rooted group lands in the slot that
                    // already holds the same tensor. A partially-tiled root (a lone split half, a
                    // tensor shared with a different-partner concat) would drag unrelated live bytes
                    // over other slices' slots, where the concat's remaining part dispatches (or the
                    // other slices' producers) overwrite them — the plain single-tensor slice is the
                    // trivial tile of itself.
                    struct SliceRef {
                        size_t off, bytes, delta, sliceIdx;
                    };
                    std::map<TensorId, std::vector<SliceRef>> byRoot;
                    for (size_t si = 0; si < slices.size(); ++si)
                    {
                        auto ro = resolveView(resolveAlias(slices[si]));
                        if (ro.first == kNoTensor || ro.first == wholeRoot || ro.first == whole || !memberOk(ro.first, flat) || offs[si] < ro.second)
                        {
                            continue;
                        }
                        byRoot[ro.first].push_back({ro.second, sliceBytes[si], offs[si] - ro.second, si});
                    }
                    for (auto &kv: byRoot)
                    {
                        const TensorId r     = kv.first;
                        auto          &tiles = kv.second;
                        const size_t   delta = tiles.front().delta;
                        bool           ok    = true;
                        for (const SliceRef &s: tiles)
                        {
                            ok = ok && s.delta == delta;
                        }
                        std::sort(tiles.begin(), tiles.end(), [](const SliceRef &a, const SliceRef &b) {
                            return a.off < b.off;
                        });
                        // Tiles must be disjoint and in-range; fullTile = they cover r's every byte.
                        size_t prevEnd  = 0;
                        bool   fullTile = true;
                        for (const SliceRef &s: tiles)
                        {
                            if (s.off < prevEnd)
                            {
                                ok = false;
                                break;
                            }
                            fullTile = fullTile && s.off == prevEnd;
                            prevEnd  = s.off + s.bytes;
                        }
                        ok       = ok && prevEnd <= actBytes(r);
                        fullTile = fullTile && prevEnd == actBytes(r);
                        ok       = ok && delta + actBytes(r) <= actBytes(whole);
                        if (ok && !fullTile)
                        {
                            // Liveness rescue for a PARTIALLY tiled root: the uncovered ranges of r's
                            // extent hold group members this concat does not re-demand, and the
                            // arena slots covering those ranges belong to OTHER slices, whose
                            // producers will overwrite the bytes. The re-root is still exact when
                            // every such member is fully read BEFORE the earliest of those producers
                            // runs (the classic ShuffleNet stride-1 block: the passthrough half is
                            // consumed by the branch convs long before the branch output lands in
                            // its slot). A member read at-or-after any overlapping writer refuses
                            // the root — including the writer itself reading the member (an
                            // intra-dispatch overlap can never be proven safe).
                            std::vector<std::pair<size_t, size_t>> uncovered;
                            {
                                size_t cur = 0;
                                for (const SliceRef &s: tiles)
                                {
                                    if (s.off > cur)
                                    {
                                        uncovered.push_back({cur, s.off});
                                    }
                                    cur = s.off + s.bytes;
                                }
                                if (cur < actBytes(r))
                                {
                                    uncovered.push_back({cur, actBytes(r)});
                                }
                            }
                            std::set<size_t> tileIdx;
                            for (const SliceRef &s: tiles)
                            {
                                tileIdx.insert(s.sliceIdx);
                            }
                            // Group members = r itself plus every linked tensor resolving to r.
                            std::vector<std::pair<TensorId, std::pair<size_t, size_t>>> members;
                            members.push_back({r, {0, actBytes(r)}});
                            for (const auto &link: viewOf)
                            {
                                auto ro2 = resolveView(link.first);
                                if (ro2.first == r)
                                {
                                    members.push_back({link.first, {ro2.second, ro2.second + actBytes(link.first)}});
                                }
                            }
                            const int neverWrites = (int) idx.size(); // past every position
                            for (const auto &m: members)
                            {
                                if (!ok)
                                {
                                    break;
                                }
                                const size_t mA = m.second.first, mB = m.second.second;
                                bool         touchesUncovered = false;
                                for (const auto &u: uncovered)
                                {
                                    if (mA < u.second && u.first < mB)
                                    {
                                        touchesUncovered = true;
                                        break;
                                    }
                                }
                                auto rd = lastReadPos.find(m.first);
                                if (!touchesUncovered || rd == lastReadPos.end())
                                {
                                    continue; // outside every clobbered range, or never read at all
                                }
                                // Earliest producer among NON-TILE slices whose slots overlap the
                                // member's arena range (a tile slot holds the member's own bytes and
                                // is written by r's original producers, never a clobber).
                                const size_t aA = delta + mA, aB = delta + mB;
                                int          earliestWriter = neverWrites;
                                for (size_t sj = 0; sj < slices.size(); ++sj)
                                {
                                    if (tileIdx.count(sj) || offs[sj] >= aB || aA >= offs[sj] + sliceBytes[sj])
                                    {
                                        continue;
                                    }
                                    auto pp        = producerPos.find(slices[sj]);
                                    earliestWriter = std::min(earliestWriter, pp == producerPos.end() ? neverWrites : pp->second);
                                }
                                ok = rd->second < earliestWriter;
                            }
                        }
                        // Claims cover exactly this concat's tiles: a rescued root's uncovered ranges
                        // are legitimately overwritten by the other slices once the members there are
                        // dead, so only the tile slots are exclusive-writer ranges.
                        for (const SliceRef &s: tiles)
                        {
                            ok = ok && claimWrite(whole, delta + s.off, delta + s.off + s.bytes);
                        }
                        if (!ok)
                        {
                            continue;
                        }
                        viewOf[r] = {whole, delta};
                        viewSlices += (int) tiles.size();
                        linked += tiles.size();
                    }
                } else
                {
                    for (size_t si = 0; si < slices.size(); ++si)
                    {
                        // A split output is a read-only alias of bytes the parent's own writers
                        // produce, so it claims no write interval. The parent must not be a session
                        // input: dma-buf zero-copy IO may rebind a session input's buffer at run
                        // time, which would orphan any views into the replaced allocation.
                        TensorId       out    = slices[si];
                        const TensorId parent = resolveAlias(whole);
                        if (!memberOk(out, flat) || viewOf.count(out) || out == wholeRoot || graphInputs_.count(parent) || offs[si] + actBytes(out) > actBytes(parent))
                        {
                            continue;
                        }
                        viewOf[out] = {parent, offs[si]};
                        ++viewSlices;
                        ++linked;
                    }
                }
                if (linked > 0)
                {
                    ++viewSites;
                    if (linked == slices.size())
                    {
                        fullyElided_.insert(ni); // record() emits nothing: skip its hazard bookkeeping
                    }
                }
            }
        }
        std::set<TensorId> viewRoots; // arena tensors whose buffer must accept sub-buffer views
        for (auto &kv: viewOf)
        {
            viewRoots.insert(resolveView(kv.first).first);
        }
        if (!viewOf.empty())
        {
            VKNN_INFO << "zero-copy slices: " << viewSlices << " view(s) into " << viewRoots.size() << " arena(s) across " << viewSites << " Concat/Split node(s)";
        }
        for (TensorId tid: acts)
        {
            // storeFp32 tensors get a dedicated buffer (never pooled): the liveness pool aliases by
            // byte size only, so a 4-byte tensor must not share a slot sized for 2-byte neighbours.
            bool internal = producedHere.count(tid) && !readBack.count(tid) && !g.tensors[tid].storeFp32;
            if (internal || viewOf.count(tid))
            {
                continue; // pooled below, or backed by a sub-buffer view created after the pool
            }
            auto pref = readBack.count(tid) ? vk::MemPref::kReadback : vk::MemPref::kAuto;
            buffers_[tid] = std::make_shared<vk::Buffer>(be_->ctx(), actBytes(tid), pref, 0, /*zeroInit=*/true, /*allowSubBufferViews=*/viewRoots.count(tid) != 0);
            // int8 KV cache: the fp16 per-row scale side buffer rides next to the payload. Zero
            // scales dequantize to 0, matching the zero-initialized fp16 cache the link path
            // starts from.
            auto kvqIt = kvqCaches_.find(tid);
            if (kvqIt != kvqCaches_.end())
            {
                kvqIt->second.scales = std::make_shared<vk::Buffer>(be_->ctx(), (size_t) kvqIt->second.rows * kKvQuantScaleBytes, vk::MemPref::kAuto, 0, /*zeroInit=*/true);
            }
        }
        // [firstPos,lastPos] of each internal tensor within this segment's execution order
        std::map<TensorId, int> firstPos, lastPos;
        auto                    touch = [&](TensorId t, int k) {
            t = resolveAlias(t);      // an aliased tensor lives in its root's buffer; extend the root's span
            t = resolveView(t).first; // a view member lives inside its arena; extend the ARENA's span
            if (t == kNoTensor || !producedHere.count(t) || readBack.count(t) || g.tensors[t].storeFp32)
            {
                return; // dedicated (storeFp32) and boundary tensors are not pooled
            }
            if (!firstPos.count(t))
            {
                firstPos[t] = k;
            }
            lastPos[t] = k;
        };
        for (int k = 0; k < (int) idx.size(); ++k)
        {
            const Node &nd = g.nodes[idx[k]];
            for (TensorId in: nd.inputs)
            {
                touch(in, k);
            }
            touch(nd.fusedResidual, k);
            for (TensorId o: nd.outputs)
            {
                touch(o, k);
            }
        }
        std::vector<TensorId> order;
        order.reserve(firstPos.size());
        for (auto &kv: firstPos)
        {
            order.push_back(kv.first);
        }
        std::sort(order.begin(), order.end(), [&](TensorId a, TensorId b) {
            return firstPos[a] < firstPos[b];
        });
        struct Slot {
            std::shared_ptr<vk::Buffer> buf;
            size_t                      cap;
            int                         deadAt;
            bool                        viewable = false; // allocated without the dedicated hint; may host views
        };
        std::vector<Slot> busy, freeSlots;
        for (TensorId tid: order)
        {
            int p = firstPos[tid];
            for (size_t i = 0; i < busy.size();)
            {
                if (busy[i].deadAt < p)
                {
                    freeSlots.push_back(busy[i]);
                    busy[i] = busy.back();
                    busy.pop_back();
                } else
                {
                    ++i;
                }
            }
            // Best-fit: reuse the smallest freed slot that still fits, so a large freed buffer is kept
            // available for a later large tensor instead of being spent (and grown) on a small one. A
            // reused slot keeps its existing (larger-or-equal) capacity; only a miss allocates anew.
            // A view-hosting arena may only reuse a VIEWABLE slot (one allocated without the
            // dedicated-memory hint, which cannot legally bind sub-buffer views); other tensors reuse
            // any slot. Arena slots re-enter the pool as viewable, so consecutive arenas share memory.
            size_t     need       = actBytes(tid);
            const bool hostsViews = viewRoots.count(tid) != 0;
            int        best       = -1;
            for (size_t i = 0; i < freeSlots.size(); ++i)
            {
                if ((!hostsViews || freeSlots[i].viewable) && freeSlots[i].cap >= need && (best < 0 || freeSlots[i].cap < freeSlots[best].cap))
                {
                    best = (int) i;
                }
            }
            Slot s;
            if (best >= 0)
            {
                s               = freeSlots[best];
                freeSlots[best] = freeSlots.back();
                freeSlots.pop_back();
            } else
            {
                s.buf      = std::make_shared<vk::Buffer>(be_->ctx(), need, vk::MemPref::kAuto, 0, /*zeroInit=*/true, /*allowSubBufferViews=*/hostsViews);
                s.cap      = need;
                s.viewable = hostsViews;
            }
            s.deadAt      = lastPos[tid];
            buffers_[tid] = s.buf;
            busy.push_back(s);
        }
        // Materialize the zero-copy views now that every arena has its buffer. A member that cannot
        // bind (a driver alignment/padding constraint — the target GPUs report 4-byte buffer
        // alignment, so none in practice) falls back to a plain buffer of its own; record() then
        // keeps the dispatching path for that slice, so a refused view is a lost optimization, never
        // an error.
        bool anyViewFallback = false;
        for (auto &kv: viewOf)
        {
            const TensorId              member     = kv.first;
            const auto                  rootAndOff = resolveView(member);
            std::shared_ptr<vk::Buffer> made;
            auto                        rit = rootAndOff.first == kNoTensor ? buffers_.end() : buffers_.find(rootAndOff.first);
            if (rit != buffers_.end())
            {
                try
                { made = std::make_shared<vk::Buffer>(be_->ctx(), rit->second, rootAndOff.second, actBytes(member)); } catch (const Error &e)
                { VKNN_WARN << "zero-copy view fallback for '" << g.tensors[member].name << "': " << e.what(); }
            }
            if (!made)
            {
                made            = std::make_shared<vk::Buffer>(be_->ctx(), actBytes(member), vk::MemPref::kAuto, 0, /*zeroInit=*/true);
                anyViewFallback = true;
            }
            buffers_[member] = made;
        }
        if (anyViewFallback)
        {
            // A fallen-back member dispatches after all (its record()-side identity check fails), so
            // its node must keep full hazard bookkeeping; dropping the whole elision set is the
            // simple safe answer for a case the target devices never hit.
            fullyElided_.clear();
            transferFillNodes_.clear();
        }
        // Point each aliased pure-copy output at its root's buffer (the root is dedicated- or pool-
        // allocated above); record() then skips the copy since src and dst resolve to the same buffer.
        for (auto &kv: aliasRoot)
        {
            auto it = buffers_.find(resolveAlias(kv.second));
            if (it != buffers_.end())
            {
                buffers_[kv.first] = it->second;
            }
        }

        // Flat-geometry view-eligibility diagnostic (opt-in: --debug-segments). Classifies each flat
        // Slice/Concat/Transpose as offset-view eligible (its output is one contiguous sub-range of the
        // input), strided (needs a gather), or a Concat disjoint write. Emits greppable [rc-diag] lines
        // that per-node profile ms attributes against. Read-only; no behaviour change.
        if (cfg_.debugSegments)
        {
            auto formatShape = [](const Shape &s) {
                std::string out = "[";
                for (size_t i = 0; i < s.size(); ++i)
                {
                    out += (i ? "," : "") + std::to_string(s[i]);
                }
                return out + "]";
            };
            int sliceViews = 0, sliceStrided = 0, concatCount = 0, transposeIdentity = 0, transposeStrided = 0;
            for (int ni: idx)
            {
                const Node &nd = g.nodes[ni];
                if (nd.outputs.empty() || nd.outputs[0] == kNoTensor || !g.desc(nd.outputs[0]).gpuFlat)
                {
                    continue;
                }
                if (nd.type == OpType::Slice)
                {
                    Shape in = g.desc(nd.inputs[0]).shape, out = g.desc(nd.outputs[0]).shape;
                    int   rank     = (int) in.size();
                    auto  steps    = readI64Param(g, nd, "steps", 4);
                    bool  unitStep = true;
                    for (auto s: steps)
                    {
                        if (s != 1)
                        {
                            unitStep = false;
                        }
                    }
                    int slicedAx = -1, nSliced = 0;
                    for (int ax = 0; ax < rank && ax < (int) out.size(); ++ax)
                    {
                        if (out[ax] != in[ax])
                        {
                            slicedAx = ax;
                            nSliced++;
                        }
                    }
                    bool contig = unitStep && nSliced <= 1 && slicedAx >= 0;
                    for (int k = 0; k < slicedAx; ++k)
                    {
                        if (in[k] != 1)
                        {
                            contig = false; // an outer dim >1 makes the slice several disjoint chunks
                        }
                    }
                    (contig ? sliceViews : sliceStrided)++;
                    VKNN_INFO << "[rc-diag] Slice " << (contig ? "VIEW " : "strd ") << nd.name << " in" << formatShape(in) << "->out" << formatShape(out) << " ax=" << slicedAx << " step1=" << unitStep << " " << actBytes(nd.outputs[0]) << "B";
                } else if (nd.type == OpType::Concat)
                {
                    concatCount++;
                    VKNN_INFO << "[rc-diag] Concat " << nd.name << " axis=" << nd.attr.geti("axis", 1) << " nin=" << nd.inputs.size() << " " << actBytes(nd.outputs[0]) << "B";
                } else if (nd.type == OpType::Transpose)
                {
                    const auto &perm  = nd.attr.getints("perm");
                    bool        ident = true;
                    for (size_t k = 0; k < perm.size(); ++k)
                    {
                        if (perm[k] != (int64_t) k)
                        {
                            ident = false; // a real permutation needs strided reads (Stage B)
                        }
                    }
                    (ident ? transposeIdentity : transposeStrided)++;
                    VKNN_INFO << "[rc-diag] Transpose " << (ident ? "VIEW " : "strd ") << nd.name << " " << actBytes(nd.outputs[0]) << "B";
                }
            }
            VKNN_INFO << "[rc-diag] SUMMARY Slice: " << sliceViews << " view / " << sliceStrided << " strided | Concat: " << concatCount << " | Transpose: " << transposeIdentity << " ident / " << transposeStrided << " strided";
        }

        // 2) build env + ops; prepare uploads weights.
        env_.backend  = be_;
        env_.ctx      = &be_->ctx();
        env_.runner   = &be_->runner();
        env_.tuning   = cfg.tuning;
        env_.winograd = (Mode) cfg.hint(Hint::Winograd, (int) Mode::Auto);
        // Load-time device resolution: the flat family's workgroup width comes from exact caps
        // here, once, and rides VkOpEnv so pipelines and dispatch math share one value.
        env_.flatLocalSize = flat::flatLocalSizeFor(env_.ctx->caps());
        env_.convLocalSize = flat::laneWidthFor(env_.ctx->caps(), flat::kConvFamilyLaneWidth);
        env_.graph         = &g;
        env_.config        = &cfg;
        env_.useFp16       = useFp16_;
        env_.baseFp16      = useFp16_; // segment-wide precision; useFp16_ is overridden per-node below for storeFp32 nodes
        // per-model weight-cache namespace: FNV-1a over the whole graph (same for every segment of this
        // model, distinct across models) so a shared cache directory can't return another model's weights.
        {
            uint64_t h   = kFnvOffsetBasis;
            auto     mix = [&](const std::string &s) {
                for (char c: s)
                {
                    h ^= (uint8_t) c;
                    h *= kFnvPrime;
                }
            };
            for (const auto &nd: g.nodes)
            {
                mix(nd.name);
                mix(opTypeName(nd.type));
            }
            mix(std::to_string(g.nodes.size()));
            char buf[20];
            snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
            env_.modelTag = buf;
        }
        { // per-GPU autotune namespace: vendor/device/driver identify the kernel-timing target.
            const auto &caps = be_->ctx().caps();
            char        tag[40];
            snprintf(tag, sizeof(tag), "%04x%04x-%08x", caps.vendorID, caps.deviceID, caps.driverVersion);
            env_.gpuTag = tag;
        }
        // Const-folding can leave an initializer as a spatial op's ACTIVATION input[0] (e.g. a baked
        // image constant fed through Cast->Resize). Such ops read input[0] via env.devBuf(), which
        // returns null for initializers — they are otherwise consumed only as weights via operandBuf.
        // Materialize any such constant into an activation buffer packed in its assigned layout so every
        // devBuf-reading op finds a valid buffer. operandBuf consumers are unaffected: they test
        // isInitializer first and upload their own flat copy, ignoring this entry.
        {
            auto materialize = [&](TensorId t) {
                if (t == kNoTensor || !g.isInitializer(t) || buffers_.count(t))
                {
                    return;
                }
                const auto        &td   = g.tensors[t];
                std::vector<float> vals = initFloats(g, t);
                int64_t            n    = numElements(td.shape);
                bool               fp16 = !td.storeFp32 && useFp16_;
                auto               buf  = std::make_shared<vk::Buffer>(be_->ctx(), actBytes(t), vk::MemPref::kAuto, 0, /*zeroInit=*/true);
                if (td.gpuFlat)
                {
                    if (fp16)
                    {
                        boundary::packFlatFp16(vals.data(), reinterpret_cast<fp16_t *>(buf->host()), n, 1);
                    } else
                    {
                        std::memcpy(buf->host(), vals.data(), (size_t) n * 4);
                    }
                } else
                {
                    boundary::packNc4(vals.data(), buf->host(), NCHW::from(td.shape), fp16, 1);
                }
                buffers_[t] = buf;
            };
            for (int ni: idx)
            {
                const Node &nd = g.nodes[ni];
                if (!nd.inputs.empty())
                {
                    materialize(nd.inputs[0]);
                }
                materialize(nd.fusedResidual);
            }
        }
        // Load + validate the model cache now that the model hash is known, then hand the primed
        // pipeline + weight caches to the env. loadCache is idempotent across this model's segments.
        be_->loadCache(cfg, env_.modelTag);
        env_.cache   = be_->pipelineCache();
        env_.weights = be_->weightCache();
        env_.devBuf  = [this](TensorId t) -> vk::Buffer  *{
            auto it = buffers_.find(t);
            return it == buffers_.end() ? nullptr : it->second.get();
        };
        // Non-zero exactly for the tensors this segment allocated with a virtualized row stride, so
        // the producing and consuming kernels read the layout decision off the allocation itself.
        env_.rowPad = [this](TensorId t) -> int64_t {
            auto it = rowPad_.find(t);
            return it == rowPad_.end() ? 0 : it->second;
        };
        // Non-null exactly for the int8 KV-cache tensors of THIS segment: FusedAttention keys its
        // kvq kernel choice off this resolver, so the kernels always agree with the allocation.
        env_.kvqScale = [this](TensorId t) -> vk::Buffer * {
            auto it = kvqCaches_.find(t);
            return it == kvqCaches_.end() ? nullptr : it->second.scales.get();
        };
        // `g` outlives every prepare() below (it is the bucket's owned graph), so the hook can drop
        // uploaded weight payloads as the ops consume them. Session frees whatever survives.
        env_.releaseInitializer = cfg.freeWeightsAfterUpload ? std::function<void(TensorId)>([&g](TensorId t) {
            auto it = g.initializers.find(t);
            if (it == g.initializers.end())
            {
                return;
            }
            // A mmap-backed view costs no heap — clearing it frees nothing but drops the ability
            // to re-read the blob. In a multi-bucket .vxm several buckets view the SAME mapped
            // blob (a weight shared by the prefill and decode plans), and each bucket's segment
            // build re-uploads it; clearing bucket 0's view would leave a later bucket's flat
            // upload with no bytes. Only OWNED (heap) payloads are worth reclaiming here.
            if (it->second.bytes.viewed())
            {
                return;
            }
            it->second.bytes.clear();
            it->second.bytes.shrink_to_fit();
        }) :
                                                               nullptr;
        // Memo scoped to this segment's graph; the ops themselves own the buffers, so a weak handle
        // keeps the allocation count identical to the pre-memo path.
        flatWeightByTensor_.clear();
        env_.lookupFlatWeight = [this](TensorId t) -> std::shared_ptr<vk::Buffer> {
            auto it = flatWeightByTensor_.find(t);
            return it == flatWeightByTensor_.end() ? nullptr : it->second.lock();
        };
        env_.rememberFlatWeight = [this](TensorId t, std::shared_ptr<vk::Buffer> buffer) {
            flatWeightByTensor_[t] = buffer;
        };
        for (int ni: idx)
        {
            auto op = VkOpRegistry::instance().create(g.nodes[ni].type);
            if (!op)
            {
                throw Error(Status::Unsupported, std::string("no Vulkan kernel for ") + opTypeName(g.nodes[ni].type));
            }
            // A storeFp32 node (its output kept in fp32) selects its fp32 kernel variant + uploads its
            // weights fp32; ConvertDtype reads the precision per tensor and ignores this.
            env_.useFp16 = nodeFp32(g.nodes[ni]) ? false : useFp16_;
            op->prepare(g.nodes[ni], env_);
            ops_.push_back(std::move(op));
        }
        env_.useFp16 = useFp16_;

        // 3) timestamp query pool (2 per node). Only when profiling - the extra writes + the implicit
        //    barriers around them aren't free, and we don't want them on the hot path.
        if (be_->ctx().caps().timestampSupported && cfg.profile)
        {
            VkQueryPoolCreateInfo qi {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
            qi.queryCount = (uint32_t) (idx.size() * 2);
            vkCreateQueryPool(be_->ctx().device(), &qi, nullptr, &queryPool_);
        }
        // Chunk begin/end timestamps for Config::timingSummary: 2 queries per command-buffer
        // chunk, written at each chunk's head/tail, read after the run's last fence. Never
        // coexists with the profiler pool - profiling forces a single chunk and per-op
        // barriers, which would perturb exactly the boundaries measured here.
        if (be_->ctx().caps().timestampSupported && cfg.timingSummary && !cfg.profile)
        {
            VkQueryPoolCreateInfo qi {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
            qi.queryCount = kMaxTimedChunks * 2;
            vkCreateQueryPool(be_->ctx().device(), &qi, nullptr, &chunkPool_);
        }

        // 4) pre-record the command buffer for the static graph.
        record();

        // The buffer count is reported against the device's allocation limit, not on its own: each
        // buffer costs one vkAllocateMemory, so a graph can run out of ALLOCATIONS while the heap
        // is far from full, and the bare count gives no way to see that coming.
        VKNN_INFO << "vk memory after segment build: live " << (vk::Buffer::liveBytes() >> 20) << " MB / " << vk::Buffer::liveCount() << " buffers (peak " << (vk::Buffer::peakBytes() >> 20) << " MB / " << vk::Buffer::peakCount() << " of "
                  << env_.ctx->caps().maxMemoryAllocationCount << " allowed, host rss " << hostRssMb() << " MB)";
    }

    size_t VulkanSegment::hostRssMb() {
#ifdef __linux__
        if (FILE *f = fopen("/proc/self/statm", "r"))
        {
            long pages = 0, rss = 0;
            if (fscanf(f, "%ld %ld", &pages, &rss) == 2)
            {
                fclose(f);
                return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE) >> 20;
            }
            fclose(f);
        }
#endif
        return 0;
    }

    VulkanSegment::~VulkanSegment() {
        if (cfg_.timingSummary && stat_.runs > 0)
        {
            const double n = (double) stat_.runs;
            VKNN_INFO << "segment summary (" << nodeIdx.size() << " nodes, " << recordedDispatches_ << " dispatches, " << cmds_.size() << " chunk(s), " << stat_.runs << " run(s)) avg ms/run: pack=" << stat_.packMs / n << " submitCall=" << stat_.submitCallMs / n << " fenceWait=" << stat_.fenceWaitMs / n << " gpuBusy=" << stat_.gpuBusyMs / n << " gpuGap=" << stat_.gpuGapMs / n << " unpack=" << stat_.unpackMs / n;
        }
        if (!cmds_.empty())
        {
            // Command buffers were allocated from the backend's shared runner pool (not owned per
            // segment), so destroying the segment must free them back or a re-planned/dynamic-shape
            // run leaks one buffer per chunk into that pool for the backend's life. be_ (and thus the
            // pool) outlives the segment -- the query-pool destroys below rely on the same fact.
            vkFreeCommandBuffers(be_->ctx().device(), be_->runner().pool(), (uint32_t) cmds_.size(), cmds_.data());
            cmds_.clear();
        }
        if (chunkPool_)
        {
            vkDestroyQueryPool(be_->ctx().device(), chunkPool_, nullptr);
        }
        if (queryPool_)
        {
            vkDestroyQueryPool(be_->ctx().device(), queryPool_, nullptr);
        }
    }

    bool VulkanSegment::nodeFp32(const Node &nd) const {
        return !nd.outputs.empty() && nd.outputs[0] != kNoTensor && g_.tensors[nd.outputs[0]].storeFp32;
    }

    void VulkanSegment::record() {
        // Recorded-dispatch accounting for this pass. Every dispatch the ops, boundary converts,
        // link copies, chain feedback, and argmax epilogues record below is counted; the per-node
        // share is attributed around each op's record() call. A re-record restarts the tally, so
        // what it reports is always the CURRENT command stream.
        DispatchTally &tally = env_.ctx->dispatchTally();
        tally.beginRun(nodeIdx.size());
        // Per-chunk begin/end timestamps (Config::timingSummary): each chunk resets and writes
        // its own query pair, so a re-recorded buffer stays self-contained. Chunks past the
        // pool capacity execute untimed; gpuBusy/gpuGap then undercount rather than misindex.
        uint32_t timedChunk   = 0;
        auto     chunkTsBegin = [&] {
            if (chunkPool_ && timedChunk < kMaxTimedChunks)
            {
                vkCmdResetQueryPool(cmd_, chunkPool_, timedChunk * 2, 2);
                vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, chunkPool_, timedChunk * 2);
            }
        };
        auto chunkTsEnd = [&] {
            if (chunkPool_ && timedChunk < kMaxTimedChunks)
            {
                vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, chunkPool_, timedChunk * 2 + 1);
                ++timedChunk;
            }
        };
        cmd_ = be_->runner().allocate();
        be_->runner().begin(cmd_);
        if (queryPool_)
        {
            vkCmdResetQueryPool(cmd_, queryPool_, 0, (uint32_t) (nodeIdx.size() * 2));
        }
        chunkTsBegin();
        // A decode chain records chainSteps_ iterations of the whole body below into one
        // command-buffer sequence; a forced chunk split at each iteration boundary lets a run
        // submit any prefix of iterations (chainActiveSteps_). The single-iteration recording
        // (no chain configured, or decodeChainSteps 1) is the degenerate one-pass loop.
        const int recordedSteps = chainConfigured_ ? chainSteps_ : 1;
        if (recordedSteps > 1)
        {
            for (const auto &kv: convert_)
            {
                if (!kv.second.isInput)
                {
                    // An output convert records once after the LAST iteration, which a chunk-
                    // prefix run would skip; no chained caller binds declared-format outputs.
                    throw Error(Status::Unsupported, "a decode-chained segment cannot bind a declared-format zero-copy output");
                }
            }
        }
        iterationFirstChunk_.assign((size_t) recordedSteps, 0);
        for (int step = 0; step < recordedSteps; ++step)
        {
            iterationFirstChunk_[(size_t) step] = (uint32_t) cmds_.size();
            // Chain state feedback: iteration `step` consumes the previous iteration's argmax index
            // as its token id, position basePosition + step, and a mask with one more valid slot —
            // each computed by the host pack's exact rules, so the chained stream is bit-identical
            // to the single-step loop. Iteration 0 consumes the host-provided inputs unchanged.
            if (step > 0)
            {
                struct ChainFeedbackPC {
                    uint32_t stepIndex, maskValidSlots, tokenElemFp16, positionElemFp16, maskElemFp16;
                };
                if (!chainFeedbackPipe_)
                {
                    chainFeedbackPipe_ = std::make_unique<vk::ComputePipeline>(be_->ctx(), "chain_feedback", 5, sizeof(ChainFeedbackPC), std::vector<uint32_t> {},
                                                                               env_.cache ? env_.cache->handle() : VK_NULL_HANDLE);
                }
                ChainFeedbackPC pc {(uint32_t) step, (uint32_t) (numElements(g_.tensors[chain_.maskInput].shape) - 1), boundaryElemBytes(chain_.tokenInput) == 2 ? 1u : 0u,
                                    boundaryElemBytes(chain_.positionInput) == 2 ? 1u : 0u, boundaryElemBytes(chain_.maskInput) == 2 ? 1u : 0u};
                chainFeedbackPipe_->dispatch(cmd_,
                                             {argMaxResults_[chain_.argMaxOutput]->handle(), buffers_[chain_.tokenInput]->handle(),
                                              buffers_[chain_.positionInput]->handle(), buffers_[chain_.maskInput]->handle(), chainStateBuf_->handle()},
                                             &pc, sizeof(pc), 1);
            }
            // Device-resident links: fold each linked output's PREVIOUS-run (or previous-iteration)
            // values into its linked input, per this iteration's range set in the host-updated
            // ranges SSBO, before anything else executes. The barrier orders the copies against
            // both hazards below: nodes reading the destination (the fold must land first) and
            // nodes rewriting the source (the copy must read the old values).
            if (!residentLinks_.empty())
            {
                struct LinkCopyPC {
                    int      srcC, srcH, srcW, dstC, dstH, dstW, srcFmt, dstFmt;
                    uint32_t rangeWordBase;
                };
                // Mirrors link_copy_kvq.comp's push_constant block.
                struct LinkCopyKvqPC {
                    int      headDim;
                    uint32_t rangeWordBase;
                };
                for (const ResidentLink &link: residentLinks_)
                {
                    if (link.kvq)
                    {
                        // Quantizing fold (Hint::KvCacheQuant): the fp16 present rows encode into the
                        // int8 cache payload + the per-row fp16 scale buffer. Same ranges SSBO and
                        // per-iteration set addressing as the bit copy.
                        if (!linkPipeKvq_)
                        {
                            linkPipeKvq_ = std::make_unique<vk::ComputePipeline>(be_->ctx(), "link_copy_kvq", 4, sizeof(LinkCopyKvqPC), std::vector<uint32_t> {},
                                                                                 env_.cache ? env_.cache->handle() : VK_NULL_HANDLE);
                        }
                        const KvqCache &cache = kvqCaches_.at(link.dst);
                        LinkCopyKvqPC   pc {(int) cache.headDim, (uint32_t) step * (2u + link.capacity * 3u)};
                        linkPipeKvq_->dispatch(cmd_, {buffers_[link.src]->handle(), buffers_[link.dst]->handle(), link.rangesBuf->handle(), cache.scales->handle()}, &pc, sizeof(pc), kKvqLinkCopyGroups);
                        continue;
                    }
                    const bool fp16 = boundaryElemBytes(link.src) == 2;
                    auto      &pipe = fp16 ? linkPipeFp16_ : linkPipeFp32_;
                    if (!pipe)
                    {
                        pipe = std::make_unique<vk::ComputePipeline>(be_->ctx(), fp16 ? "link_copy_fp16" : "link_copy", 3, sizeof(LinkCopyPC), std::vector<uint32_t> {},
                                                                     env_.cache ? env_.cache->handle() : VK_NULL_HANDLE);
                    }
                    NCHW       srcShape = NCHW::from(g_.tensors[link.src].shape);
                    NCHW       dstShape = NCHW::from(g_.tensors[link.dst].shape);
                    LinkCopyPC pc {(int) srcShape.c,
                                   (int) srcShape.h,
                                   (int) srcShape.w,
                                   (int) dstShape.c,
                                   (int) dstShape.h,
                                   (int) dstShape.w,
                                   g_.desc(link.src).gpuFlat ? 0 : 2,
                                   g_.desc(link.dst).gpuFlat ? 0 : 2,
                                   (uint32_t) step * (2u + link.capacity * 3u)};
                    pipe->dispatch(cmd_, {buffers_[link.src]->handle(), buffers_[link.dst]->handle(), link.rangesBuf->handle()}, &pc, sizeof(pc), kLinkCopyGroups);
                }
            }
            if (step > 0 || !residentLinks_.empty())
            {
                vk::computeBarrier(*env_.ctx, cmd_);
            }
            // Declared-format zero-copy inputs: convert each caller dma-buf (declared layout/dtype) into
            // this segment's device-native boundary buffer, then a barrier before the ops read it.
            // Iteration 0 only: later chain iterations take their inputs from the feedback dispatch,
            // which a re-run convert would overwrite with the stale iteration-0 bytes.
            if (step == 0)
            {
                bool any = false;
                for (const auto &kv: convert_)
                {
                    if (!kv.second.isInput)
                    {
                        continue;
                    }
                    const ConvertBinding &c = kv.second;
                    if (!conv_)
                    {
                        conv_ = std::make_unique<BoundaryConvert>();
                    }
                    conv_->record(cmd_, *env_.ctx, env_.cache, c.imported.get(), buffers_[kv.first].get(), c.shape, c.declFmt, c.declDtype, c.devFmt, c.devDtype);
                    any = true;
                }
                if (any)
                {
                    vk::computeBarrier(*env_.ctx, cmd_);
                }
            }
            auto isCopy = [&](int idx) {
                const Node &nn = g_.nodes[idx];
                OpType      t  = nn.type;
                // A flat split is a compute dispatch (flat_gather); the NC4HW4 split is a buffer copy.
                if (t == OpType::Split)
                {
                    return nn.outputs.empty() || nn.outputs[0] == kNoTensor || !g_.desc(nn.outputs[0]).gpuFlat;
                }
                // A zero-copy Pad records vkCmdFillBuffer for its pad ranges — transfer-stage writes.
                if (t == OpType::Pad)
                {
                    return transferFillNodes_.count(idx) != 0;
                }
                // Reshape/Flatten/Squeeze/Unsqueeze/Cast are vkCmdCopyBuffer (transfer-stage writes).
                return t == OpType::Reshape || t == OpType::Flatten || t == OpType::Squeeze || t == OpType::Unsqueeze || t == OpType::Cast;
            };
            // Precise barriers: each activation tensor has a single writer, so only a read-after-write
            // needs a barrier. Emit one before an op only when it reads a tensor written since the last
            // barrier, letting independent ops (e.g. the parallel branches of an Inception module, or a
            // residual block's downsample and conv1) run without draining the GPU between them. When
            // profiling, keep a barrier after every op so the per-op timestamps aren't polluted by overlap.
            const bool perOpBarrier = (queryPool_ != VK_NULL_HANDLE);
            // Hazard tracking is at the BUFFER level, not the tensor level: the liveness planner aliases
            // multiple tensors onto one buffer, so a node that writes a reused buffer has a
            // write-after-read hazard against the previous occupant that a tensor-level check would miss.
            // For non-aliased buffers this reduces to per-tensor read-after-write (single writer per
            // buffer), so independent-op overlap (Inception/YOLO) is preserved.
            std::set<vk::Buffer *> writtenBufs, readBufs;
            auto                   bufOf = [&](TensorId t) -> vk::Buffer                   *{
                if (t == kNoTensor)
                {
                    return nullptr;
                }
                auto it = buffers_.find(t);
                return it == buffers_.end() ? nullptr : it->second.get();
            };
            // With zero-copy sub-buffer views, two distinct buffer handles can address overlapping
            // memory (a slice view and its arena, or two views of one arena), so hazard membership is a
            // (root, byte-range) overlap test rather than a handle match. Non-view buffers keep the old
            // exact semantics: their range is the whole buffer and distinct roots never overlap, so
            // disjoint slices of one arena (parallel Inception branches writing their slots) still
            // record no barrier between them.
            auto hazard = [](const std::set<vk::Buffer *> &s, vk::Buffer *b) {
                for (vk::Buffer *a: s)
                {
                    if (a == b)
                    {
                        return true;
                    }
                    if (a->hazardRoot() == b->hazardRoot())
                    {
                        const size_t a0 = a->rootOffset(), b0 = b->rootOffset();
                        if (a0 < b0 + b->bytes() && b0 < a0 + a->bytes())
                        {
                            return true;
                        }
                    }
                }
                return false;
            };
            bool copySinceBarrier = false;
            // Push-descriptor writes a node records = one per bound storage buffer. A fused
            // pointwise kernel binds the plan SSBO plus its operand slots and the kPwMaxOuts extra
            // output streams on top of its own inputs/outputs; a plain op binds just those. A
            // STANDALONE unit carries kPwMaxOperands slots, an epilogue the narrower
            // kPwEpilogueMaxOperands, since that header is inlined into every producer kernel. Concat dispatches once per concatenated part, re-binding
            // the full set each time. Accumulated per command buffer, this drives the
            // maxSubmitBindings split below.
            auto bindEstimate = [&](const Node &nd) -> int {
                const int pwSlots = nd.type == OpType::FusedPointwise ? kPwMaxOperands : kPwEpilogueMaxOperands;
                int       pwExtra = (nd.type == OpType::FusedPointwise || nd.attr.has("pw_steps")) ? 1 + pwSlots + kPwMaxOuts : 0;
                if (nd.type == OpType::Concat)
                {
                    return (int) pwCoreInputs(nd) * (2 + pwExtra);
                }
                return (int) nd.inputs.size() + (int) nd.outputs.size() + pwExtra;
            };
            int nodesSinceSplit = 0, bindsSinceSplit = 0;
            int fullBarriers = 0, execBarriers = 0; // recorded-barrier tally for the segment INFO line
            for (size_t k = 0; k < nodeIdx.size(); ++k)
            {
                const Node &node = g_.nodes[nodeIdx[k]];
                // A fully-elided zero-copy node records no commands and touches no memory: its data
                // hazards ride the real producers/consumers through the shared arena ranges, so it
                // neither needs a barrier nor marks reads/writes (a phantom mark would insert a
                // redundant pipeline drain at every elided site).
                const bool elided      = fullyElided_.count(nodeIdx[k]) != 0;
                bool       needBarrier = perOpBarrier && !elided; // full memory barrier (RAW/WAW: data must become visible)
                bool       needWarOnly = false;                   // execution-only barrier (WAR on a reused pool slot: order, no data)
                if (!needBarrier && !elided)
                {
                    for (TensorId in: node.inputs) // read-after-write
                    {
                        if (vk::Buffer *b = bufOf(in))
                        {
                            if (hazard(writtenBufs, b))
                            {
                                needBarrier = true;
                                break;
                            }
                        }
                    }
                    if (!needBarrier)
                    {
                        if (vk::Buffer *b = bufOf(node.fusedResidual))
                        {
                            if (hazard(writtenBufs, b))
                            {
                                needBarrier = true;
                            }
                        }
                    }
                    if (!needBarrier)
                    {
                        for (TensorId o: node.outputs) // write-after-write (unflushed writer) / write-after-read (reused buffer)
                        {
                            if (vk::Buffer *b = bufOf(o))
                            {
                                if (hazard(writtenBufs, b))
                                {
                                    needBarrier = true;
                                    break;
                                }
                                if (hazard(readBufs, b))
                                {
                                    needWarOnly = true; // upgrade to full below if a copy is involved
                                }
                            }
                        }
                    }
                    // A WAR against a vkCmdCopyBuffer read (or ahead of a copy write) crosses the
                    // transfer stage, which the compute-only execution barrier does not order.
                    if (needWarOnly && (copySinceBarrier || isCopy(nodeIdx[k])))
                    {
                        needBarrier = true;
                    }
                }
                if (needBarrier)
                {
                    if (copySinceBarrier || isCopy(nodeIdx[k]))
                    {
                        vk::transferBarrier(*env_.ctx, cmd_);
                    } else
                    {
                        vk::computeBarrier(*env_.ctx, cmd_);
                    }
                    ++fullBarriers;
                    writtenBufs.clear();
                    readBufs.clear();
                    copySinceBarrier = false;
                } else if (needWarOnly)
                {
                    // Orders this node's write after every recorded read; earlier writes stay in
                    // writtenBufs because nothing here made them available or visible.
                    vk::executionBarrier(*env_.ctx, cmd_);
                    ++execBarriers;
                    readBufs.clear();
                }
                if (queryPool_)
                {
                    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, (uint32_t) (k * 2));
                }
                env_.useFp16 = nodeFp32(node) ? false : useFp16_; // match the variant chosen in prepare()
                tally.openNode(k);
                ops_[k]->record(cmd_, node, env_);
                tally.closeNode(); // a decode chain re-enters this loop per iteration; counts accumulate
                if (queryPool_)
                {
                    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, (uint32_t) (k * 2 + 1));
                }
                if (!elided)
                {
                    for (TensorId in: node.inputs)
                    {
                        if (vk::Buffer *b = bufOf(in))
                        {
                            readBufs.insert(b);
                        }
                    }
                    if (vk::Buffer *b = bufOf(node.fusedResidual))
                    {
                        readBufs.insert(b);
                    }
                    for (TensorId o: node.outputs)
                    {
                        if (vk::Buffer *b = bufOf(o))
                        {
                            writtenBufs.insert(b);
                        }
                    }
                    if (isCopy(nodeIdx[k]))
                    {
                        copySinceBarrier = true;
                    }
                }
                // Split the segment into multiple command buffers so no single batch (a) runs long
                // enough to trip the GPU watchdog (a ~20s submit on this driver gets reset silently,
                // zeroing the unexecuted tail) or (b) records more push-descriptor writes than the
                // driver holds (maxSubmitBindings; a newer driver corrupts the recording past its cap).
                // The explicit barrier at each chunk tail keeps buffer reuse correct across the
                // boundary (the chunks of one run are submitted together). Only when not profiling.
                nodesSinceSplit++;
                bindsSinceSplit += bindEstimate(node);
                const int  chunkNodes = cfg_.maxSubmitNodes, chunkBinds = cfg_.maxSubmitBindings;
                const bool splitNodes = chunkNodes > 0 && nodesSinceSplit >= chunkNodes;
                const bool splitBinds = chunkBinds > 0 && bindsSinceSplit >= chunkBinds;
                if (!queryPool_ && (splitNodes || splitBinds) && k + 1 < nodeIdx.size())
                {
                    // Each chunk is its own vkQueueSubmit + fence wait (see run()). The fence is a
                    // full barrier, so a chunk's writes are complete and visible before the next chunk
                    // is submitted and buffer reuse stays correct across the boundary. Batching the
                    // chunks into one submit and ordering them with a vkCmdPipelineBarrier at the
                    // chunk tail is faster but unsafe: a submit spanning several chunks runs long
                    // enough for the driver to reset it and zero the tail (the watchdog the chunking
                    // exists to avoid), so a heavily-chunked model then produces nondeterministic
                    // output. Keeping chunks as separate submits keeps every submit short.
                    chunkTsEnd();
                    be_->runner().end(cmd_);
                    cmds_.push_back(cmd_);
                    cmd_ = be_->runner().allocate();
                    be_->runner().begin(cmd_);
                    chunkTsBegin();
                    writtenBufs.clear();
                    readBufs.clear();
                    copySinceBarrier = false;
                    nodesSinceSplit  = 0;
                    bindsSinceSplit  = 0;
                }
            }
            if (fullBarriers + execBarriers > 0)
            {
                VKNN_INFO << "segment barriers: " << fullBarriers << " full + " << execBarriers << " execution-only over " << nodeIdx.size() << " node(s)";
            }
            // Final barrier so the segment outputs are complete + visible before the host reads them.
            if (copySinceBarrier)
            {
                vk::transferBarrier(*env_.ctx, cmd_);
            } else
            {
                vk::computeBarrier(*env_.ctx, cmd_);
            }
            // Registered output argmax epilogues: one single-workgroup dispatch per output, reading
            // the finished boundary buffer and writing {index, value} into its per-iteration result
            // slot. Rides the same submission; the final barrier above makes the outputs visible to it.
            for (TensorId argMaxTid: argMaxOutputs_)
            {
                struct ArgMaxPC {
                    uint32_t elemCount, resultSlot;
                };
                const bool fp16 = boundaryElemBytes(argMaxTid) == 2;
                auto      &pipe = fp16 ? argMaxPipeFp16_ : argMaxPipeFp32_;
                if (!pipe)
                {
                    pipe = std::make_unique<vk::ComputePipeline>(be_->ctx(), fp16 ? "argmax_flat_fp16" : "argmax_flat", 2, sizeof(ArgMaxPC),
                                                                 std::vector<uint32_t> {flat::laneWidthPow2For(be_->ctx().caps(), flat::kFlatLocalSize)},
                                                                 env_.cache ? env_.cache->handle() : VK_NULL_HANDLE);
                }
                ArgMaxPC pc {(uint32_t) numElements(g_.tensors[argMaxTid].shape), (uint32_t) step};
                pipe->dispatch(cmd_, {buffers_[argMaxTid]->handle(), argMaxResults_[argMaxTid]->handle()}, &pc, sizeof(pc), 1);
            }
            // Iteration boundary: end the chunk so a run can submit any iteration prefix, with the
            // tail barrier ordering this iteration's argmax/node writes against the next
            // iteration's feedback, link copies, and node reads (the same contract as a
            // maxSubmitNodes split; each chunk submits with its own fence).
            if (step + 1 < recordedSteps)
            {
                vk::transferBarrier(*env_.ctx, cmd_);
                chunkTsEnd();
                be_->runner().end(cmd_);
                cmds_.push_back(cmd_);
                cmd_ = be_->runner().allocate();
                be_->runner().begin(cmd_);
                chunkTsBegin();
            }
        }
        // Declared-format zero-copy outputs: convert the device-native boundary buffer into each
        // caller dma-buf (declared layout/dtype), then a barrier before the host reads it.
        {
            bool any = false;
            for (const auto &kv: convert_)
            {
                if (kv.second.isInput)
                {
                    continue;
                }
                const ConvertBinding &c = kv.second;
                if (!conv_)
                {
                    conv_ = std::make_unique<BoundaryConvert>();
                }
                conv_->record(cmd_, *env_.ctx, env_.cache, buffers_[kv.first].get(), c.imported.get(), c.shape, c.devFmt, c.devDtype, c.declFmt, c.declDtype);
                any = true;
            }
            if (any)
            {
                vk::computeBarrier(*env_.ctx, cmd_);
            }
        }
        chunkTsEnd();
        be_->runner().end(cmd_);
        cmds_.push_back(cmd_);
        timedChunks_     = timedChunk;
        recorded_        = true;
        recordedConvert_ = convert_;
        // Snapshot the tally into the segment: the tally belongs to the device context, so the next
        // segment (or plan bucket) to record on it restarts the run and overwrites the table. The
        // profile the run reports must be THIS segment's recording.
        nodeDispatches_.assign(nodeIdx.size(), 0);
        for (size_t k = 0; k < nodeIdx.size(); ++k)
        {
            nodeDispatches_[k] = tally.nodeDispatches(k);
        }
        recordedDispatches_ = tally.runTotal();
        // The real per-run dispatch count, alongside the barrier tally above. It runs well above
        // the node count: one op records several dispatches (split-K partial + reduce, Winograd's
        // three passes, fused attention's partial + combine), and the boundary converts, resident
        // link copies, chain feedback, and argmax epilogues dispatch outside any node. The op share
        // also lands per node in the Config::profile table (OpRecord::dispatches).
        const uint64_t nodeShare = tally.nodeTotal();
        VKNN_INFO << "segment dispatches: " << recordedDispatches_ << " over " << nodeIdx.size() << " node(s) (" << nodeShare << " from nodes + " << (recordedDispatches_ - nodeShare) << " boundary/epilogue)";
    }

    void VulkanSegment::run(ExecContext &ctx) {
        const bool timing = cfg_.timing;
        auto       now    = [] {
            return std::chrono::high_resolution_clock::now();
        };
        auto t0 = now();
        // --- zero-copy: bind a caller dma-buf fd (rt.dmaBufFd) as the boundary GPU buffer so the GPU
        //     reads/writes it directly. Re-record the command buffer when the bound-buffer set changes;
        //     the imported buffer is cached by fd, so a reused dma-buf re-records once. Import failure
        //     keeps the pooled buffer (the copy path). Boundary I/O buffers are dedicated (not
        //     pool-aliased), so swapping them is safe.
        {
            bool reRecord = false;
            convert_.clear();
            auto rebind = [&](TensorId tid, bool isInput) {
                auto bit = buffers_.find(tid);
                if (bit == buffers_.end())
                {
                    return;
                }
                if (!origBoundary_.count(tid))
                {
                    origBoundary_[tid] = bit->second; // snapshot the pooled boundary buffer
                }
                std::shared_ptr<vk::Buffer> want = origBoundary_[tid];
                RtTensor                   &rt   = ctx.t(tid);
                int                         fd   = rt.dmaBufFd;
                if (fd >= 0 && kvqCaches_.count(tid))
                {
                    // An int8 KV cache cannot bind a caller dma-buf: the fd holds fp16 rows, the
                    // resident buffer int8 codes + side scales. The host seed path quantizes instead.
                    fd = -1;
                }
                if (fd >= 0)
                {
                    bool         flat    = g_.desc(tid).gpuFlat;
                    TensorFormat devFmt  = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
                    DType        devDt   = useFp16_ ? DType::Float16 : DType::Float32;
                    TensorFormat declFmt = rt.dmaBufFormat;
                    DType        declDt  = rt.dmaBufDtype;
                    bool         direct  = declFmt == TensorFormat::Auto || (declFmt == devFmt && declDt == devDt);
                    NCHW         x       = NCHW::from(rt.shape.empty() ? g_.tensors[tid].shape : rt.shape);
                    // Import sized for what the dma-buf actually holds: the device-native bytes for a
                    // direct bind, the declared-format bytes for a convert. Re-import when this
                    // tensor's fd or size changes.
                    size_t needB = direct ? origBoundary_[tid]->bytes() : (size_t) (formatElems(declFmt, x) * dtypeSize(declDt));
                    if (needB > 0)
                    {
                        uint64_t  id    = dmaBufId(fd);
                        Imported &imp   = imported_[tid];
                        bool      stale = !imp.buf || imp.bytes != needB || (id != 0 ? imp.id != id : imp.fd != fd);
                        if (stale)
                        {
                            std::unique_ptr<vk::Buffer> b = vk::Buffer::importDmaBufFd(be_->ctx(), fd, needB);
                            imp                           = {id, fd, needB, std::shared_ptr<vk::Buffer>(std::move(b))};
                            if (!imp.buf)
                            {
                                // No dma-buf import on this device: zero-copy can't be honored. The
                                // pooled buffer holds no caller data, so the result for this input is
                                // invalid — surface it rather than read silently undefined memory.
                                VKNN_WARN_THROTTLE("zerocopy-import-fail", 1) << "dma-buf import failed for '" << g_.tensors[tid].name << "' (device lacks dma-buf import); zero-copy unavailable";
                            }
                        }
                        if (imp.buf)
                        {
                            if (direct)
                            {
                                want = imp.buf; // declared == device-native: bind the fd directly
                            } else
                            {
                                // declared != device-native: keep the pooled boundary buffer; the GPU
                                // converts between the imported buffer and it (recorded in record()).
                                ConvertBinding cb;
                                cb.imported   = imp.buf;
                                cb.isInput    = isInput;
                                cb.shape      = x;
                                cb.declFmt    = declFmt;
                                cb.declDtype  = declDt;
                                cb.devFmt     = devFmt;
                                cb.devDtype   = devDt;
                                convert_[tid] = cb;
                            }
                        }
                    }
                }
                if (bit->second != want)
                {
                    bit->second = want;
                    reRecord    = true;
                }
            };
            for (TensorId tid: boundaryInputs)
            {
                rebind(tid, true);
            }
            for (TensorId tid: boundaryOutputs)
            {
                rebind(tid, false);
            }
            // Default-path GPU image conversion: for each 8-bit graph input NOT bound to a dma-buf this
            // run (and not already handled by the dma-buf rebind), stand up a persistent staging buffer
            // and a boundary_convert(staging[declared] -> boundary[device-native]) so the raw caller
            // bytes are converted on the GPU. The staging buffer's stable identity keeps this a one-time
            // re-record. Skipped when a dma-buf fd is present (zero-copy wins) or the graph is not
            // whole-GPU (ioGpuConvert off -> host packToBuffer path).
            if (ioGpuConvert)
            {
                for (TensorId tid: boundaryInputs)
                {
                    if (!graphInputs_.count(tid) || buffers_.find(tid) == buffers_.end())
                    {
                        continue;
                    }
                    RtTensor &rt = ctx.t(tid);
                    if (rt.dmaBufFd >= 0 || convert_.count(tid))
                    {
                        continue; // zero-copy dma-buf (direct or its own convert) takes precedence
                    }
                    // A raw 8-bit image input (session-stashed as its declared dtype) and a rank-4
                    // fp32 input both convert on the GPU here: the declared bytes upload to a staging
                    // buffer and boundary_convert produces the device-native boundary, skipping the host
                    // pack (uint8->fp32->fp16 for images, fp32->fp16+NC4HW4 for the rank-4 float case).
                    // boundary_convert mirrors packToBuffer's index math and RTE fp16 rounding, so a
                    // converted input is byte-identical to the host pack. The rank-4 gate keeps the win
                    // on the large image inputs it targets (a [N,C,H,W] feature map) and off the tiny
                    // per-token fp32 boundaries (inputs_embeds [1,S,H], 1-D/2-D masks and index vectors,
                    // scalars) where it is a no-win; Int8/Int32/Int64 have no boundary_convert variant.
                    const std::vector<int64_t> &inShape   = rt.shape.empty() ? g_.tensors[tid].shape : rt.shape;
                    const bool                  fp32Image = rt.dtype == DType::Float32 && inShape.size() == 4;
                    if (rt.dtype != DType::UInt8 && rt.dtype != DType::Int8 && !fp32Image)
                    {
                        continue;
                    }
                    if (linkedInputs_.count(tid))
                    {
                        // A linked input's device buffer IS its resident state (updated in place across
                        // runs); a per-submit staging convert would overwrite it. Keep the host path.
                        continue;
                    }
                    if (kvqCaches_.count(tid))
                    {
                        // An int8 KV cache has no boundary_convert variant (int8 payload + side
                        // scales); the host seed path quantizes instead.
                        continue;
                    }
                    bool         flat    = g_.desc(tid).gpuFlat;
                    TensorFormat devFmt  = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
                    DType        devDt   = (useFp16_ && !g_.tensors[tid].storeFp32) ? DType::Float16 : DType::Float32;
                    TensorFormat declFmt = TensorFormat::NCHW; // caller image layout
                    DType        declDt  = rt.dtype;
                    NCHW         x       = NCHW::from(rt.shape.empty() ? g_.tensors[tid].shape : rt.shape);
                    auto        &st      = stagingIn_[tid];
                    size_t       need    = (size_t) (formatElems(declFmt, x) * dtypeSize(declDt));
                    if (need == 0)
                    {
                        continue; // a zero-dim boundary input has no bytes to stage; vkCreateBuffer(size=0) is invalid -> keep the host path
                    }
                    if (!st || st->bytes() != need)
                    {
                        st = std::make_shared<vk::Buffer>(be_->ctx(), need, vk::MemPref::kAuto);
                    }
                    ConvertBinding cb;
                    cb.imported   = st;
                    cb.isInput    = true;
                    cb.shape      = x;
                    cb.declFmt    = declFmt;
                    cb.declDtype  = declDt;
                    cb.devFmt     = devFmt;
                    cb.devDtype   = devDt;
                    convert_[tid] = cb;
                }
                // The same on the way out. A graph output the device holds as fp16 but the caller
                // declared fp32 is downloaded today by reading the device mapping element by element
                // and widening each one; on a 2.8 MB output that is the single largest host cost of a
                // run. A boundary_convert into a HOST_CACHED staging buffer leaves the host a plain
                // memcpy. fp16 -> fp32 is exact and NCHW -> NCHW is identity, so the bytes are the
                // ones the host loop produced.
                for (TensorId tid: boundaryOutputs)
                {
                    if (!graphOutputs_.count(tid) || buffers_.find(tid) == buffers_.end())
                    {
                        continue;
                    }
                    // Every case the download path handles some other way keeps that way: a
                    // zero-copy fd, a resident link, an on-device argmax, a row-sliced readback (it
                    // wants ONE row, not the converted whole), an int8 KV cache, and the NC4HW4
                    // outputs, which do not take the flat download branch this replaces.
                    RtTensor &rt = ctx.t(tid);
                    if (rt.dmaBufFd >= 0 || convert_.count(tid) || linkedOutputs_.count(tid) || argMaxOutputs_.count(tid) || rowSelectOutputs_.count(tid) ||
                        kvqCaches_.count(tid) || !g_.desc(tid).gpuFlat)
                    {
                        continue;
                    }
                    const bool deviceFp16 = useFp16_ && !g_.tensors[tid].storeFp32;
                    if (!deviceFp16 || g_.tensors[tid].dtype != DType::Float32)
                    {
                        continue; // no widening to do: the download is already a straight copy
                    }
                    const std::vector<int64_t> &outShape = rt.shape.empty() ? g_.tensors[tid].shape : rt.shape;
                    NCHW                        y        = NCHW::from(outShape);
                    const size_t                need     = (size_t) (formatElems(TensorFormat::NCHW, y) * dtypeSize(DType::Float32));
                    if (need == 0)
                    {
                        continue; // a zero-dim output has no bytes to stage; vkCreateBuffer(size=0) is invalid
                    }
                    auto &st = stagingOut_[tid];
                    if (!st || st->bytes() != need)
                    {
                        st = std::make_shared<vk::Buffer>(be_->ctx(), need, vk::MemPref::kReadback);
                    }
                    ConvertBinding cb;
                    cb.imported   = st;
                    cb.isInput    = false;
                    cb.shape      = y;
                    cb.declFmt    = TensorFormat::NCHW;
                    cb.declDtype  = DType::Float32;
                    cb.devFmt     = TensorFormat::NCHW;
                    cb.devDtype   = DType::Float16;
                    convert_[tid] = cb;
                }
            }
            if (!sameConvert(convert_, recordedConvert_))
            {
                reRecord = true;
            }
            if (linksChanged_)
            {
                // The resident-link set (or a ranges buffer's identity) changed since the last
                // recording; the command stream must pick up the new link_copy dispatches.
                linksChanged_ = false;
                reRecord      = true;
            }
            if (argMaxChanged_)
            {
                // The registered reduction set changed; the recording must append its epilogue.
                argMaxChanged_ = false;
                reRecord       = true;
            }
            if (chainChanged_)
            {
                // The decode-chain configuration changed; the recording must carry the chained
                // iteration sequence (or drop back to the single-iteration stream).
                chainChanged_ = false;
                reRecord      = true;
            }
            if (reRecord)
            {
                if (!cmds_.empty())
                {
                    vkFreeCommandBuffers(be_->ctx().device(), be_->runner().pool(), (uint32_t) cmds_.size(), cmds_.data());
                    cmds_.clear();
                }
                record();
            }
        }
        // attach boundary buffers to RtTensors (cross-segment residency) + upload inputs.
        // Each segment owns a SEPARATE buffer per tensor, so a boundary input must be (re)packed into
        // THIS segment's buffer unless that exact buffer already holds the data. Matching on the exact
        // buffer (not just rt.deviceValid) is required: a tensor produced by an earlier GPU segment is
        // deviceValid but points at that segment's buffer, so this segment must repack into its own.
        for (TensorId tid: boundaryInputs)
        {
            RtTensor &rt  = ctx.t(tid);
            auto      bit = buffers_.find(tid);
            if (bit == buffers_.end())
            {
                continue;
            }
            bool alreadyHere = rt.deviceValid && rt.device && rt.device->buffer == bit->second;
            bool flat        = g_.desc(tid).gpuFlat;
            if (!rt.device)
            {
                rt.device = std::make_shared<DeviceStorage>();
            }
            rt.device->buffer = bit->second;
            auto sit          = stagingIn_.find(tid);
            // Session may have LENT this input's bytes rather than copying them, which is valid only
            // for the staging-convert route below. Every other route reads owned host bytes, so take
            // ownership before entering the chain.
            const bool stagedInput = sit != stagingIn_.end() && convert_.count(tid);
            if (!stagedInput)
            {
                rt.materializeHostBorrow();
            }
            if (rt.dmaBufFd >= 0 && !kvqCaches_.count(tid))
            {
                // zero-copy: the GPU reads the caller's dma-buf directly (device-native bytes); no pack.
                // An int8 KV cache never binds an fd (the rebind refused the import); its host seed
                // branch below quantizes instead.
                rt.deviceValid  = true;
                rt.deviceFormat = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
            } else if (stagedInput)
            {
                // GPU image convert: raw memcpy the caller's declared bytes into the staging buffer; the
                // recorded boundary_convert dispatch turns them into the device-native boundary. No host
                // uint8->fp32->fp16 pack. The convert writes bit->second (the boundary), read by the ops.
                // The bytes are the caller's own when Session lent them (no host mirror was filled);
                // otherwise they are the mirror's. Either way this is the ONLY copy of an input.
                const uint8_t *src      = rt.hostBorrow ? rt.hostBorrow : rt.host.bytes.data();
                const size_t   srcBytes = rt.hostBorrow ? rt.hostBorrowBytes : rt.host.bytes.size();
                std::memcpy(sit->second->host(), src, std::min(sit->second->bytes(), srcBytes));
                rt.deviceValid  = true;
                rt.deviceFormat = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
            } else if (rt.hostValid && !alreadyHere && kvqCaches_.count(tid))
            {
                // int8 KV cache re-seed (the prefill -> decode hand-off): quantize the fp32 host
                // mirror into the payload + scale buffers with the host codec — the host mirror
                // itself stays float (quantize at upload, dequantize at download).
                seedKvqFromHost(tid, rt);
                rt.deviceValid  = true;
                rt.deviceFormat = TensorFormat::NCHW;
            } else if (rt.hostValid && !alreadyHere)
            {
                // The Vulkan device represents an integer tensor as its float value (index/shape ops
                // upload int64 indices decoded to float), but rt.host for an int64/int32 boundary
                // tensor holds raw integer bytes. packToBuffer reads host as fp32, so decode the
                // integer host to fp32 first; a Float32 host packs directly. Without this, an int64
                // boundary input crossing into a Vulkan segment (e.g. attention_mask when a mid-graph
                // CPU island splits the graph) is reinterpreted as fp32 and comes out ~0.
                if (rt.dtype == DType::Int64 || rt.dtype == DType::Int32)
                {
                    RtTensor f32 = rt;
                    f32.dtype    = DType::Float32;
                    int64_t n    = numElements(rt.shape);
                    f32.host.resizeElems(n, DType::Float32);
                    float *d = f32.host.f32();
                    if (rt.dtype == DType::Int64)
                    {
                        const int64_t *s = rt.host.i64();
                        for (int64_t i = 0; i < n; ++i)
                        {
                            d[i] = (float) s[i];
                        }
                    } else
                    {
                        const int32_t *s = reinterpret_cast<const int32_t *>(rt.host.bytes.data());
                        for (int64_t i = 0; i < n; ++i)
                        {
                            d[i] = (float) s[i];
                        }
                    }
                    // A storeFp32 boundary (a pinned Gather index) keeps its 4-byte fp32 buffer, so an
                    // integer index above the fp16 range is not narrowed to +inf at upload.
                    VulkanBackend::packToBuffer(bit->second.get(), f32, g_.desc(tid).storeFp32 ? false : useFp16_, flat, cpu::threadCount(&cfg_));
                } else
                {
                    VulkanBackend::packToBuffer(bit->second.get(), rt, g_.desc(tid).storeFp32 ? false : useFp16_, flat, cpu::threadCount(&cfg_));
                }
                rt.deviceValid  = true;
                rt.deviceFormat = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
            }
            if (linkedInputs_.count(tid))
            {
                // A linked input's device buffer IS its state: never bound -> the zero-initialized
                // buffer (plus the link copies) is authoritative; a caller (re)bind went through the
                // pack above because the Session stamped deviceValid=false on it. Either way the
                // residency is now valid and stays so across runs.
                rt.deviceValid  = true;
                rt.deviceFormat = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
            }
        }
        auto t1 = now();

        // One submit + fence wait per chunk. The fence fully orders each chunk against the next,
        // so results are deterministic even on a driver that does not honor a vkCmdPipelineBarrier
        // across command-buffer boundaries within one submission, and no submit runs long enough
        // to trip the GPU watchdog. A decode chain submits the chunk prefix covering its active
        // iterations.
        const bool     summarize    = cfg_.timingSummary;
        double         submitCalls  = 0;
        const uint32_t submitChunks = chunksForActiveSteps();
        double         wall         = 0;
        for (uint32_t ci = 0; ci < submitChunks; ++ci)
        {
            double sc = 0;
            wall += be_->runner().submitAndWait(cmds_[ci], summarize ? &sc : nullptr);
            submitCalls += sc;
        }
        auto t2 = now();

        // download boundary outputs to host.
        std::set<TensorId> graphOut(g_.outputs.begin(), g_.outputs.end());
        for (TensorId tid: boundaryOutputs)
        {
            auto bit = buffers_.find(tid);
            if (bit == buffers_.end())
            {
                continue;
            }
            RtTensor &rt   = ctx.t(tid);
            bool      flat = g_.desc(tid).gpuFlat;
            if (!rt.device)
            {
                rt.device = std::make_shared<DeviceStorage>();
            }
            rt.device->buffer = bit->second;
            rt.deviceValid    = true;
            rt.deviceFormat   = flat ? TensorFormat::NCHW : TensorFormat::NC4HW4;
            if (linkedOutputs_.count(tid))
            {
                // A linked output stays device-resident: no download. The stale host copy is
                // invalidated; readResident() unpacks on demand.
                rt.hostValid = false;
                continue;
            }
            if (argMaxOutputs_.count(tid))
            {
                // Reduced on-device: the host reads the 8-byte result via readOutputArgMax();
                // the full vector never downloads and any stale host copy is invalid.
                rt.hostValid = false;
                continue;
            }
            if (rt.dmaBufFd < 0)
            {
                bool deviceFp16 = useFp16_ && !g_.tensors[tid].storeFp32;
                auto rowIt      = rowSelectOutputs_.find(tid);
                if (rowIt != rowSelectOutputs_.end() && flat && graphOut.count(tid))
                {
                    // setOutputRow: download only row `rowIt->second` of the flat [.., R, V] output
                    // (V elements from offset row*V), skipping the R-1 unread rows — the prefill logits
                    // case (one 256xV logits matrix, only the last real token's row consumed). rt.shape
                    // stays [.., R, V]; the Session emits the sliced io.shape from its own record.
                    const int64_t V     = rt.shape.empty() ? 0 : rt.shape.back();
                    const int64_t total = numElements(rt.shape);
                    const int64_t nRows = V > 0 ? total / V : 0;
                    const int64_t row   = rowIt->second;
                    if (V > 0 && row >= 0 && row < nRows)
                    {
                        VulkanBackend::downloadFlatOutput(bit->second.get(), rt, deviceFp16, g_.tensors[tid].dtype, cpu::threadCount(&cfg_), row * V, V);
                    } else
                    { // out-of-range selection: fall back to the full readback rather than miscopy
                        VulkanBackend::downloadFlatOutput(bit->second.get(), rt, deviceFp16, g_.tensors[tid].dtype, cpu::threadCount(&cfg_));
                    }
                } else if (auto sout = stagingOut_.find(tid); sout != stagingOut_.end() && convert_.count(tid))
                {
                    // The recorded boundary_convert already widened this output into the staging
                    // buffer in the declared dtype, so the download is a memcpy out of HOST_CACHED
                    // memory rather than a per-element read of the device mapping.
                    const int64_t n = numElements(rt.shape);
                    rt.host.resizeElems(n, DType::Float32);
                    std::memcpy(rt.host.bytes.data(), sout->second->host(), std::min(sout->second->bytes(), rt.host.bytes.size()));
                    rt.dtype = DType::Float32;
                } else if (flat && graphOut.count(tid))
                { // terminal graph output: convert straight to the declared dtype (skip fp32 round trip)
                    VulkanBackend::downloadFlatOutput(bit->second.get(), rt, deviceFp16, g_.tensors[tid].dtype, cpu::threadCount(&cfg_));
                } else
                {
                    VulkanBackend::unpackFromBuffer(bit->second.get(), rt, deviceFp16, flat, cpu::threadCount(&cfg_));
                }
            }
            // else: the GPU wrote device-native bytes straight into the caller's dma-buf; caller reads it.
        }
        if (timing || summarize)
        {
            auto t3 = now();
            auto ms = [&](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            if (summarize)
            {
                stat_.runs += 1;
                stat_.packMs += ms(t0, t1);
                stat_.submitCallMs += submitCalls;
                stat_.fenceWaitMs += wall - submitCalls;
                stat_.unpackMs += ms(t2, t3);
                // All fences above have signalled, so the chunk timestamps are available. Only
                // the chunks this run actually submitted wrote their query pairs (a chain
                // prefix skips the tail chunks; their queries would wait forever).
                const uint32_t timedThisRun = std::min(timedChunks_, submitChunks);
                if (chunkPool_ && timedThisRun > 0)
                {
                    std::vector<uint64_t> ts((size_t) timedThisRun * 2, 0);
                    vkGetQueryPoolResults(be_->ctx().device(), chunkPool_, 0, (uint32_t) ts.size(), ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                    const double period = be_->ctx().caps().timestampPeriod;
                    for (uint32_t c = 0; c < timedThisRun; ++c)
                    {
                        stat_.gpuBusyMs += (double) (ts[c * 2 + 1] - ts[c * 2]) * period / 1e6;
                        if (c > 0)
                        {
                            stat_.gpuGapMs += (double) (ts[c * 2] - ts[(c - 1) * 2 + 1]) * period / 1e6;
                        }
                    }
                }
            }
            if (timing)
            {
                VKNN_INFO << "timing: pack=" << ms(t0, t1) << "ms submit+gpu=" << wall << "ms unpack=" << ms(t2, t3) << "ms";
            }
        }

        // Config::dumpTensors targeted dump: write the named tensors (dedicated buffers) to disk for
        // diffing.
        if (!dumpTids_.empty())
        {
            ::mkdir(cfg_.layerDumpDir.c_str(), 0755);
            for (TensorId tid: dumpTids_)
            {
                auto bit = buffers_.find(tid);
                if (bit == buffers_.end())
                {
                    continue;
                }
                RtTensor &rt = ctx.t(tid);
                if (kvqCaches_.count(tid))
                {
                    // int8 KV cache: dump the dequantized values, not the raw codes.
                    rt.shape = rt.shape.empty() ? g_.tensors[tid].shape : rt.shape;
                    dequantKvqToHost(tid, rt);
                } else
                {
                    VulkanBackend::unpackFromBuffer(bit->second.get(), rt, useFp16_ && !g_.tensors[tid].storeFp32, g_.desc(tid).gpuFlat);
                }
                std::string nm = g_.tensors[tid].name;
                for (char &c: nm)
                {
                    if (c == '/' || c == ':')
                    {
                        c = '_';
                    }
                }
                FILE *f = fopen((cfg_.layerDumpDir + "/" + nm + ".bin").c_str(), "wb");
                if (f)
                {
                    fwrite(rt.host.bytes.data(), 1, rt.host.bytes.size(), f);
                    fclose(f);
                }
            }
        }
        // layer-dump: bring every activation back to host for per-layer diffing.
        if (ctx.config && ctx.config->layerDump)
        {
            for (auto &kv: buffers_)
            {
                RtTensor &rt = ctx.t(kv.first);
                if (g_.isInitializer(kv.first))
                {
                    continue;
                }
                if (kvqCaches_.count(kv.first))
                {
                    // int8 KV cache: the raw payload is codes, not fp16 words — dump the
                    // dequantized values instead.
                    rt.shape = rt.shape.empty() ? g_.tensors[kv.first].shape : rt.shape;
                    dequantKvqToHost(kv.first, rt);
                    continue;
                }
                VulkanBackend::unpackFromBuffer(kv.second.get(), rt, useFp16_ && !g_.tensors[kv.first].storeFp32, g_.desc(kv.first).gpuFlat);
            }
        }

        // profiler: per-node GPU time from timestamps + dispatch dims.
        if (ctx.profiler && ctx.profiler->enabled() && queryPool_)
        {
            std::vector<uint64_t> ts(nodeIdx.size() * 2, 0);
            vkGetQueryPoolResults(be_->ctx().device(), queryPool_, 0, (uint32_t) ts.size(), ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            double period = be_->ctx().caps().timestampPeriod;
            for (size_t k = 0; k < nodeIdx.size(); ++k)
            {
                const Node &node = g_.nodes[nodeIdx[k]];
                OpRecord    r;
                r.name    = node.name;
                r.type    = node.type;
                r.backend = "Vulkan";
                r.gpuMs   = (double) (ts[k * 2 + 1] - ts[k * 2]) * period / 1e6;
                r.cpuMs   = 0;
                // What record() attributed to this node: 0 for an op the planner elided to a
                // zero-copy view or lowered to a buffer copy, >1 for a multi-pass kernel.
                r.dispatches = k < nodeDispatches_.size() ? nodeDispatches_[k] : 0;
                ctx.profiler->add(r);
            }
            // GPU span (first dispatch start -> last dispatch end) vs the CPU-side submit wall: the
            // difference is barrier bubbles + submit/fence latency, not kernel work. The span is
            // also the run's ELAPSED GPU time, which the per-node records cannot give: the GPU
            // overlaps consecutive nodes, so their intervals overlap and summing them over-reports.
            double span = (double) (ts.back() - ts.front()) * period / 1e6;
            ctx.profiler->addGpuSpanMs(span);
            VKNN_INFO << "gpu span=" << span << "ms  submit-wall=" << wall << "ms  (gap = overhead)";
        }
    }

    Status VulkanSegment::addResidentLink(TensorId sourceOutput, TensorId destInput, std::string &whyNot) {
        auto srcIt = buffers_.find(sourceOutput);
        auto dstIt = buffers_.find(destInput);
        if (srcIt == buffers_.end() || dstIt == buffers_.end())
        {
            whyNot = "the segment holds no device buffer for a linked tensor";
            return Status::InvalidArgument;
        }
        const bool kvqDst = kvqCaches_.count(destInput) != 0;
        if (kvqDst)
        {
            // int8 KV-cache destination: the fold quantizes instead of bit-copying, so the element
            // sizes legitimately differ (fp16 source rows, int8 payload). The quantizing fold is
            // defined on flat fp16 sources only — the eligibility rule guarantees both, so a miss
            // here is a real wiring error worth surfacing.
            if (boundaryElemBytes(sourceOutput) != 2 || !g_.desc(sourceOutput).gpuFlat || !g_.desc(destInput).gpuFlat)
            {
                whyNot = "the int8 KV-cache fold into '" + g_.tensors[destInput].name + "' needs a flat fp16 present source; '" + g_.tensors[sourceOutput].name + "' is not one";
                return Status::InvalidArgument;
            }
        } else if (boundaryElemBytes(sourceOutput) != boundaryElemBytes(destInput))
        {
            whyNot = "device element size differs between '" + g_.tensors[sourceOutput].name + "' and '" + g_.tensors[destInput].name + "' (fp16 vs pinned-fp32 storage); the raw copy would misalign";
            return Status::InvalidArgument;
        }
        if (srcIt->second == dstIt->second)
        {
            whyNot = "'" + g_.tensors[sourceOutput].name + "' and '" + g_.tensors[destInput].name + "' share one device buffer; an in-place ranged copy would race";
            return Status::InvalidArgument;
        }
        for (const ResidentLink &link: residentLinks_)
        {
            if (link.src == sourceOutput && link.dst == destInput)
            {
                return Status::Ok; // already registered; ranges arrive via setResidentLinkRanges
            }
        }
        ResidentLink link;
        link.src       = sourceOutput;
        link.dst       = destInput;
        link.kvq       = kvqDst;
        link.capacity  = kLinkInitialRangeCapacity;
        link.rangesBuf = std::make_shared<vk::Buffer>(be_->ctx(), linkRangesBufferBytes(link.capacity), vk::MemPref::kAuto, 0, /*zeroInit=*/true);
        residentLinks_.push_back(std::move(link));
        linkedInputs_.insert(destInput);
        linkedOutputs_.insert(sourceOutput);
        linksChanged_ = true; // the next run re-records with the link_copy dispatch at the head
        return Status::Ok;
    }

    void VulkanSegment::setResidentLinkRangeSets(TensorId sourceOutput, TensorId destInput, const std::vector<std::vector<LinkRange>> &rangeSets) {
        for (ResidentLink &link: residentLinks_)
        {
            if (link.src != sourceOutput || link.dst != destInput)
            {
                continue;
            }
            if (link.kvq)
            {
                // The quantizing fold processes whole token rows (one absmax + scale per row);
                // sub-row or misaligned ranges have no defined scale semantics. The engine's own
                // fold drivers (kvFoldRanges) are always row-aligned, so a violation is a caller
                // bug surfaced hard rather than a silently mis-scaled cache.
                const int64_t headDim = kvqCaches_.at(link.dst).headDim;
                for (const std::vector<LinkRange> &ranges: rangeSets)
                {
                    for (const LinkRange &range: ranges)
                    {
                        if (range.sourceElem % headDim != 0 || range.destElem % headDim != 0 || range.count % headDim != 0)
                        {
                            throw Error(Status::InvalidArgument, "int8 KV-cache link '" + g_.tensors[link.dst].name + "': fold ranges must cover whole " + std::to_string(headDim) + "-element token rows (got source " +
                                                                     std::to_string(range.sourceElem) + ", dest " + std::to_string(range.destElem) + ", count " +
                                                                     std::to_string(range.count) + ")");
                        }
                    }
                }
            }
            uint32_t neededCapacity = 0;
            for (const std::vector<LinkRange> &ranges: rangeSets)
            {
                neededCapacity = std::max<uint32_t>(neededCapacity, (uint32_t) ranges.size());
            }
            if (neededCapacity > link.capacity)
            {
                link.capacity  = std::max<uint32_t>(link.capacity * 2, neededCapacity);
                link.rangesBuf = std::make_shared<vk::Buffer>(be_->ctx(), linkRangesBufferBytes(link.capacity), vk::MemPref::kAuto, 0, /*zeroInit=*/true);
                linksChanged_  = true; // buffer identity (and the per-set stride) changed; the recording binds the old one
            }
            // One set per chain iteration at a fixed stride of {rangeCount, totalElems} + 3
            // uint32 per range slot. Iterations past the last provided set get a zero header, so
            // their recorded copy dispatch is a no-op. The previous run's fence has signalled
            // (submitAndWait), so the GPU is not reading this buffer here.
            const size_t strideWords = 2 + (size_t) link.capacity * 3;
            uint32_t    *words       = reinterpret_cast<uint32_t *>(link.rangesBuf->host());
            for (int setIdx = 0; setIdx < chainStepsMax_; ++setIdx)
            {
                uint32_t *setWords = words + (size_t) setIdx * strideWords;
                if ((size_t) setIdx >= rangeSets.size())
                {
                    setWords[0] = 0;
                    setWords[1] = 0;
                    continue;
                }
                const std::vector<LinkRange> &ranges = rangeSets[(size_t) setIdx];
                uint32_t                      total  = 0;
                for (size_t i = 0; i < ranges.size(); ++i)
                {
                    setWords[2 + i * 3 + 0] = (uint32_t) ranges[i].sourceElem;
                    setWords[2 + i * 3 + 1] = (uint32_t) ranges[i].destElem;
                    setWords[2 + i * 3 + 2] = (uint32_t) ranges[i].count;
                    total += (uint32_t) ranges[i].count;
                }
                setWords[0] = (uint32_t) ranges.size();
                setWords[1] = total;
            }
            return;
        }
    }

    void VulkanSegment::clearResidentLinks() {
        if (residentLinks_.empty())
        {
            return;
        }
        residentLinks_.clear();
        linkedInputs_.clear();
        linkedOutputs_.clear();
        linksChanged_ = true;
    }

    bool VulkanSegment::downloadResident(TensorId id, RtTensor &rt) {
        auto bit = buffers_.find(id);
        if (bit == buffers_.end())
        {
            return false;
        }
        rt.shape = rt.shape.empty() ? g_.tensors[id].shape : rt.shape;
        if (kvqCaches_.count(id))
        {
            // int8 KV cache: the host mirror stays float — dequantize on download (the inverse of
            // the seed-path quantize), so a caller (the prefill hand-off, readResident) sees the
            // same fp32 mirror convention as the fp16 cache path.
            dequantKvqToHost(id, rt);
            return true;
        }
        VulkanBackend::unpackFromBuffer(bit->second.get(), rt, useFp16_ && !g_.tensors[id].storeFp32, g_.desc(id).gpuFlat, cpu::threadCount(&cfg_));
        return true;
    }

    void VulkanSegment::dequantKvqToHost(TensorId id, RtTensor &rt) {
        const KvqCache &cache = kvqCaches_.at(id);
        rt.host.resizeElems(cache.rows * cache.headDim, DType::Float32);
        rt.dtype = DType::Float32;
        kvDequantRows(reinterpret_cast<const int8_t *>(buffers_.at(id)->host()), reinterpret_cast<const fp16_t *>(cache.scales->host()), cache.rows, cache.headDim,
                      rt.host.f32());
        rt.hostValid = true;
    }

    void VulkanSegment::seedKvqFromHost(TensorId id, const RtTensor &rt) {
        const KvqCache &cache   = kvqCaches_.at(id);
        int8_t         *payload = reinterpret_cast<int8_t *>(buffers_.at(id)->host());
        fp16_t         *scales  = reinterpret_cast<fp16_t *>(cache.scales->host());
        if (rt.dtype == DType::Float32)
        {
            // The via-fp16 pre-round makes the seeded bytes identical to a GPU fold of the same
            // values (a mirror downloaded from an fp16 buffer is already fp16-representable, so
            // the pre-round is the identity on the production hand-off).
            kvQuantRowsFromFp32ViaFp16(reinterpret_cast<const float *>(rt.host.bytes.data()), cache.rows, cache.headDim, payload, scales);
            return;
        }
        if (rt.dtype == DType::Float16)
        {
            std::vector<float> widened((size_t) (cache.rows * cache.headDim));
            halfToFloatBulk(reinterpret_cast<const fp16_t *>(rt.host.bytes.data()), widened.data(), (int64_t) widened.size());
            kvQuantRows(widened.data(), cache.rows, cache.headDim, payload, scales);
            return;
        }
        throw Error(Status::InvalidArgument, "int8 KV cache '" + g_.tensors[id].name + "' seeded from a non-float host tensor (" + dtypeStr(rt.dtype) + ")");
    }

    Status VulkanSegment::setOutputArgMax(TensorId output, std::string &whyNot) {
        if (std::find(boundaryOutputs.begin(), boundaryOutputs.end(), output) == boundaryOutputs.end() || !buffers_.count(output))
        {
            whyNot = "'" + g_.tensors[output].name + "' is not a boundary output of this segment";
            return Status::InvalidArgument;
        }
        if (!g_.desc(output).gpuFlat)
        {
            // The epilogue indexes flat elements; an NC4HW4 boundary would report block-order
            // indices. The Session reduces the host copy instead.
            whyNot = "'" + g_.tensors[output].name + "' is not flat on the device";
            return Status::Unsupported;
        }
        if (argMaxOutputs_.insert(output).second)
        {
            // One {uint index, float value} slot per decode-chain iteration; a single-step
            // segment holds (and writes) slot 0 only.
            argMaxResults_[output] = std::make_shared<vk::Buffer>(be_->ctx(), kArgMaxResultBytes * (size_t) chainStepsMax_, vk::MemPref::kAuto, 0, /*zeroInit=*/true);
            argMaxChanged_ = true; // the next run re-records with the epilogue dispatch
        }
        return Status::Ok;
    }

    Status VulkanSegment::setOutputRow(TensorId output, int64_t row, std::string &whyNot) {
        if (row < 0)
        {
            rowSelectOutputs_.erase(output); // clear -> full readback resumes
            return Status::Ok;
        }
        if (std::find(boundaryOutputs.begin(), boundaryOutputs.end(), output) == boundaryOutputs.end() || !buffers_.count(output))
        {
            whyNot = "'" + g_.tensors[output].name + "' is not a boundary output of this segment";
            return Status::InvalidArgument;
        }
        if (!g_.desc(output).gpuFlat)
        {
            // The slice copies a contiguous flat row; an NC4HW4 boundary interleaves rows.
            whyNot = "'" + g_.tensors[output].name + "' is not flat on the device";
            return Status::Unsupported;
        }
        rowSelectOutputs_[output] = row; // consulted in the boundary-output download loop
        return Status::Ok;
    }

    bool VulkanSegment::readOutputArgMax(TensorId output, int step, int64_t &index, float &value) {
        auto it = argMaxResults_.find(output);
        if (it == argMaxResults_.end() || step < 0 || step >= chainStepsMax_)
        {
            return false;
        }
        // {uint index, float value} per iteration slot, written by the epilogue dispatch of the
        // last submitted run; the fence has signalled (submitAndWait), so the mapped read
        // is coherent.
        const uint32_t *words = reinterpret_cast<const uint32_t *>(it->second->host()) + (size_t) step * 2;
        index                 = (int64_t) words[0];
        std::memcpy(&value, &words[1], sizeof value);
        return true;
    }

    Status VulkanSegment::configureDecodeChain(TensorId tokenInput, TensorId positionInput, TensorId maskInput, TensorId argMaxOutput, int steps, std::string &whyNot) {
        if (steps < 1 || steps > chainStepsMax_)
        {
            whyNot = "chain length " + std::to_string(steps) + " is outside [1, " + std::to_string(chainStepsMax_) + "] (Config::decodeChainSteps sizes the per-iteration buffers)";
            return Status::InvalidArgument;
        }
        if (cfg_.profile)
        {
            // Profiling records per-node timestamps over a single-iteration stream (one query
            // pair per node) and forces a single chunk; a chain would misindex both.
            whyNot = "profiling and decode chains are mutually exclusive";
            return Status::Unsupported;
        }
        if (!argMaxOutputs_.count(argMaxOutput))
        {
            whyNot = "'" + g_.tensors[argMaxOutput].name + "' is not registered for the on-device argmax (setOutputArgMax first)";
            return Status::InvalidArgument;
        }
        const TensorId feedbackInputs[3] = {tokenInput, positionInput, maskInput};
        for (TensorId feedbackInput: feedbackInputs)
        {
            if (std::find(boundaryInputs.begin(), boundaryInputs.end(), feedbackInput) == boundaryInputs.end() || !buffers_.count(feedbackInput))
            {
                whyNot = "'" + g_.tensors[feedbackInput].name + "' is not a boundary input of the argmax segment";
                return Status::InvalidArgument;
            }
            // The feedback dispatch writes flat element positions. A flat buffer is trivially
            // fine; so is an NC4HW4 buffer whose NCHW view has H == W == 1 (a [1,1] id/position
            // or a [1, C+1] mask), where the channel-block interleave maps every canonical
            // element to its own index — the identity, exactly like packNc4.
            const NCHW deviceView = NCHW::from(g_.tensors[feedbackInput].shape);
            if (!g_.desc(feedbackInput).gpuFlat && (deviceView.h != 1 || deviceView.w != 1))
            {
                whyNot = "'" + g_.tensors[feedbackInput].name + "' has a non-identity device layout; the feedback dispatch writes flat elements";
                return Status::Unsupported;
            }
        }
        if (numElements(g_.tensors[tokenInput].shape) != 1 || numElements(g_.tensors[positionInput].shape) != 1)
        {
            whyNot = "the token/position inputs must hold exactly one element (a [1,1] decode step)";
            return Status::InvalidArgument;
        }
        if (numElements(g_.tensors[maskInput].shape) < 2)
        {
            whyNot = "the mask input must span the context window plus the current-token slot";
            return Status::InvalidArgument;
        }
        chain_.tokenInput    = tokenInput;
        chain_.positionInput = positionInput;
        chain_.maskInput     = maskInput;
        chain_.argMaxOutput  = argMaxOutput;
        chainSteps_          = steps;
        chainConfigured_     = true;
        chainActiveSteps_    = 1; // widened per chain via setDecodeChainWindow
        if (!chainStateBuf_)
        {
            chainStateBuf_ = std::make_shared<vk::Buffer>(be_->ctx(), sizeof(uint32_t), vk::MemPref::kAuto, 0, /*zeroInit=*/true);
        }
        chainChanged_ = true; // the next run re-records the chained command stream
        return Status::Ok;
    }

    Status VulkanSegment::setDecodeChainWindow(int64_t basePosition, int activeSteps) {
        if (!chainConfigured_)
        {
            return Status::Unsupported;
        }
        if (activeSteps < 1 || activeSteps > chainSteps_)
        {
            VKNN_ERROR << "decode chain: active steps " << activeSteps << " outside [1, " << chainSteps_ << "]";
            return Status::InvalidArgument;
        }
        // The feedback dispatch derives fp32 position values from basePosition + step with exact
        // integer arithmetic; past 2^24 a float can no longer hold the integer exactly, matching
        // the host pack's own limit.
        if (basePosition < 0 || basePosition + chainSteps_ > (int64_t) 1 << 24)
        {
            VKNN_ERROR << "decode chain: base position " << basePosition << " outside the exactly-representable range";
            return Status::InvalidArgument;
        }
        // The previous run's fence has signalled (submitAndWait), so the GPU is not
        // reading the chain-state buffer here.
        *reinterpret_cast<uint32_t *>(chainStateBuf_->host()) = (uint32_t) basePosition;
        chainActiveSteps_                                     = activeSteps;
        return Status::Ok;
    }

    uint64_t VulkanSegment::dmaBufId(int fd) {
        struct stat st;
        if (::fstat(fd, &st) != 0)
        {
            return 0;
        }
        return ((uint64_t) st.st_dev * kFnvPrime) ^ (uint64_t) st.st_ino; // (dev,inode) -> stable id
    }

    size_t VulkanSegment::linkRangesBufferBytes(uint32_t rangeCapacity) const {
        return (size_t) chainStepsMax_ * (kLinkRangeHeaderBytes + (size_t) rangeCapacity * 12);
    }

    uint32_t VulkanSegment::chunksForActiveSteps() const {
        const int recordedSteps = chainConfigured_ ? chainSteps_ : 1;
        if (chainActiveSteps_ >= recordedSteps || iterationFirstChunk_.size() < (size_t) recordedSteps)
        {
            return (uint32_t) cmds_.size();
        }
        return iterationFirstChunk_[(size_t) chainActiveSteps_];
    }

    int VulkanSegment::boundaryElemBytes(TensorId tid) const {
        return (useFp16_ && !g_.tensors[tid].storeFp32) ? 2 : 4;
    }

    bool VulkanSegment::sameConvert(const std::map<TensorId, ConvertBinding> &a, const std::map<TensorId, ConvertBinding> &b) {
        if (a.size() != b.size())
        {
            return false;
        }
        for (const auto &kv: a)
        {
            auto it = b.find(kv.first);
            if (it == b.end())
            {
                return false;
            }
            const ConvertBinding &x = kv.second, &y = it->second;
            if (x.imported.get() != y.imported.get() || x.isInput != y.isInput || x.declFmt != y.declFmt || x.declDtype != y.declDtype || x.devFmt != y.devFmt || x.devDtype != y.devDtype ||
                x.shape.n != y.shape.n || x.shape.c != y.shape.c || x.shape.h != y.shape.h || x.shape.w != y.shape.w)
            {
                return false;
            }
        }
        return true;
    }

} // namespace vknn
