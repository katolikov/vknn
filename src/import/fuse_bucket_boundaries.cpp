// Fuse cross-bucket hand-offs in a multi-graph model so a value produced by one graph and consumed by
// another stays inside one graph on the GPU instead of round-tripping through the host.
//
// Background for a reader new to VKNN: some models are shipped as several graphs kept as separate
// "buckets" in one compiled model, and a host loop chains them by copying one graph's output into the
// next graph's input. The classic case is a vision-language model:
//     input_ids --(embed graph)--> inputs_embeds --(decoder graph)--> logits
// but ANY export split this way behaves the same. When a producer graph's single output is handed to a
// different graph across the host boundary, a lone host-bounded GPU op (e.g. the embedding lookup) gets
// folded onto the CPU (the GPU round-trip costs more than the op), so the model is not fully on the GPU.
//
// This pass removes that boundary at COMPILE time. The rule is model-agnostic: whenever bucket A has a
// graph OUTPUT whose name equals a graph INPUT of bucket B, A's subgraph is copied into B so B produces
// that value itself; B then takes A's own inputs directly. A that is fully absorbed this way (all of its
// outputs are consumed by other buckets) is deleted. Buckets that are not chained by a matching
// output/input name — independent dispatch targets, or a recurrence like the KV cache whose names differ
// (`present.*` out vs `past_key_values.*` in) — are left exactly as they were.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/node.h"
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace vknn {

    namespace {
        std::vector<std::string> boundaryNames(const Graph &g, const std::vector<TensorId> &ids) {
            std::vector<std::string> out;
            out.reserve(ids.size());
            for (TensorId id: ids)
            {
                out.push_back(g.desc(id).name);
            }
            return out;
        }
        // Graph inputs/outputs are authoritative as the Graph::inputs / Graph::outputs id lists (the
        // per-tensor isInput/isOutput flags are not always restored from a compiled .vxm).
        bool inList(const std::vector<TensorId> &list, TensorId id) {
            return std::find(list.begin(), list.end(), id) != list.end();
        }
        void dropFromList(std::vector<TensorId> &list, TensorId id) {
            list.erase(std::remove(list.begin(), list.end(), id), list.end());
        }

        // Copy producer graph `src` into consumer graph `dst`, connecting src's output `boundary` (a
        // graph output of src, and currently a graph input of dst with the same name) so dst produces it
        // internally. src's own graph inputs become dst graph inputs; src's initializers/intermediates
        // are copied with fresh ids so the two id spaces never collide.
        void mergeProducerInto(const Graph &src, Graph &dst, const std::string &boundary) {
            const TensorId srcOut = src.find(boundary);
            const TensorId dstIn  = dst.find(boundary);
            if (srcOut == kNoTensor || dstIn == kNoTensor)
            {
                return;
            }
            std::map<TensorId, TensorId> remap; // src tensor id -> dst tensor id

            // The boundary tensor: dst's existing input tensor, now produced internally by the copy.
            dst.desc(dstIn).isInput = false;
            dropFromList(dst.inputs, dstIn);
            remap[srcOut] = dstIn;

            // src's graph inputs become dst graph inputs (reuse dst's tensor if it already has that name,
            // e.g. a shared `attention_mask`, so we never add a duplicate boundary input).
            for (TensorId si: src.inputs)
            {
                const std::string &nm = src.desc(si).name;
                TensorId           existing = dst.find(nm);
                if (existing != kNoTensor && inList(dst.inputs, existing))
                {
                    remap[si] = existing;
                    continue;
                }
                TensorDesc d  = src.desc(si);
                d.isOutput    = false;
                TensorId nt   = dst.addTensor(d);
                dst.inputs.push_back(nt);
                remap[si] = nt;
            }
            // Every remaining src tensor (initializers, intermediates) -> a fresh dst tensor.
            for (TensorId t = 0; t < (TensorId) src.tensors.size(); ++t)
            {
                if (remap.count(t))
                {
                    continue;
                }
                TensorDesc d = src.desc(t);
                d.isInput    = false;
                d.isOutput   = false;
                TensorId nt  = dst.addTensor(d);
                remap[t]     = nt;
                if (src.isInitializer(t))
                {
                    dst.initializers[nt] = src.initializers.at(t);
                }
            }
            // Copy src nodes with remapped ids, prepended so they run before dst's body consumes them.
            std::vector<Node> copied;
            copied.reserve(src.nodes.size());
            for (const Node &sn: src.nodes)
            {
                Node nn = sn;
                for (TensorId &in: nn.inputs)
                {
                    if (in != kNoTensor && remap.count(in))
                    {
                        in = remap.at(in);
                    }
                }
                for (TensorId &o: nn.outputs)
                {
                    if (o != kNoTensor && remap.count(o))
                    {
                        o = remap.at(o);
                    }
                }
                if (nn.fusedResidual != kNoTensor && remap.count(nn.fusedResidual))
                {
                    nn.fusedResidual = remap.at(nn.fusedResidual);
                }
                if (nn.fusedBias != kNoTensor && remap.count(nn.fusedBias))
                {
                    nn.fusedBias = remap.at(nn.fusedBias);
                }
                copied.push_back(std::move(nn));
            }
            dst.nodes.insert(dst.nodes.begin(), copied.begin(), copied.end());
        }
    } // namespace

    // Fuse every producer->consumer name hand-off across buckets; delete fully-absorbed producers.
    // Single-bucket models and non-chained buckets are left untouched.
    void fuseBucketBoundaries(std::vector<Graph> &buckets, std::vector<std::string> &names) {
        if (buckets.size() < 2)
        {
            return;
        }
        bool changed = true;
        while (changed)
        {
            changed = false;
            // For each ordered pair (producer A, consumer B), find a name that is A's output and B's input.
            for (size_t a = 0; a < buckets.size() && !changed; ++a)
            {
                const std::vector<std::string> aOut = boundaryNames(buckets[a], buckets[a].outputs);
                for (size_t b = 0; b < buckets.size(); ++b)
                {
                    if (b == a)
                    {
                        continue;
                    }
                    std::string boundary;
                    for (const std::string &on: aOut)
                    {
                        const TensorId aid = buckets[a].find(on);
                        const TensorId bid = buckets[b].find(on);
                        // Same NAME and same SHAPE => a genuine hand-off. The shape check pairs the right
                        // producer with the right consumer when a model has several buckets that share a
                        // boundary name at different token counts (e.g. a prefill and a decode graph).
                        if (bid != kNoTensor && inList(buckets[b].inputs, bid) && aid != kNoTensor
                            && buckets[a].desc(aid).shape == buckets[b].desc(bid).shape)
                        {
                            boundary = on;
                            break;
                        }
                    }
                    if (boundary.empty())
                    {
                        continue;
                    }
                    // Copy A into B across this boundary. (A snapshot is taken because merging reads A.)
                    printf("[fuse] merged bucket %zu into bucket %zu on hand-off '%s'\n", a, b, boundary.c_str());
                    Graph producer = buckets[a];
                    mergeProducerInto(producer, buckets[b], boundary);

                    // If A is now fully absorbed — every one of its outputs is consumed as some other
                    // bucket's input — delete it; otherwise leave it (it still has an independent output).
                    // We only reach here by merging A across a real hand-off, so A is a feeder. Delete it
                    // once NO remaining bucket still consumes any of its outputs (every hand-off consumer
                    // has absorbed it). A feeder that still feeds another bucket at a different shape (e.g.
                    // a decode-shape decoder not yet merged) stays until that one is merged too.
                    bool stillConsumed = false;
                    for (TensorId ao: buckets[a].outputs)
                    {
                        const std::string on = buckets[a].desc(ao).name;
                        const Shape       sh = buckets[a].desc(ao).shape;
                        for (size_t c = 0; c < buckets.size() && !stillConsumed; ++c)
                        {
                            if (c == a)
                            {
                                continue;
                            }
                            const TensorId cid = buckets[c].find(on);
                            if (cid != kNoTensor && inList(buckets[c].inputs, cid) && buckets[c].desc(cid).shape == sh)
                            {
                                stillConsumed = true;
                            }
                        }
                    }
                    if (!stillConsumed)
                    {
                        buckets.erase(buckets.begin() + a);
                        names.erase(names.begin() + a);
                    }
                    changed = true; // restart the scan: indices shifted / a new boundary may have opened
                    break;
                }
            }
        }
    }

} // namespace vknn
