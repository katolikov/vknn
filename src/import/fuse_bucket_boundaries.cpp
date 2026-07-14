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
//
// A vision-language model has one further hand-off that is NOT a name match: the image encoder's feature
// rows are written into the token embeddings at the image-placeholder positions. Which rows to overwrite
// is a run-time fact (it depends on the prompt), so that one is wired as an on-GPU ScatterND — see
// spliceImageFeatures — into an extra, image-capable copy of the decoder, leaving the original as the
// text-only path.
#include "import/passes.h"
#include "vknn/graph.h"
#include "vknn/logging.h"
#include "vknn/node.h"
#include <algorithm>
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

            // The boundary tensor: dst's existing input tensor, now produced internally by the copy. Map
            // the producer's boundary output onto it — and any other producer tensor sharing that name
            // (some exports carry an unused value-info placeholder of the same name), so no stray duplicate
            // is copied in and dst keeps exactly one tensor by that name.
            dst.desc(dstIn).isInput = false;
            dropFromList(dst.inputs, dstIn);
            for (TensorId t = 0; t < (TensorId) src.tensors.size(); ++t)
            {
                if (src.desc(t).name == boundary)
                {
                    remap[t] = dstIn;
                }
            }

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

        // A vision-language model has one more cross-graph hand-off that is NOT a plain name match: the
        // image encoder's feature rows are written into the token-embedding sequence at the positions of
        // the image placeholder tokens. Which rows to overwrite is known only at run time (from the
        // prompt), so this is a scatter, not a name fusion like the embedding lookup above. This helper
        // wires it as an on-GPU ScatterND inside the decoder: after the merged embedding lookup produces
        // the token embeddings, ScatterND overwrites the image-token rows with the image features. Two new
        // graph inputs carry the run-time data — `featureName` (the encoder's feature output, bound across
        // from the vision graph) and "image_positions" (the row each feature overwrites, computed by the
        // caller). `embeds` is the decoder's internal token-embedding tensor.
        void spliceImageFeatures(Graph &dec, TensorId embeds, const std::string &featureName,
                                 const Shape &featureShape, DType featureDtype) {
            // Route the embedding lookup's output into a private tensor so ScatterND can read it as its
            // `data` operand and write the spliced result back into the original `embeds` the decoder body
            // already consumes (so no downstream node needs rewiring).
            TensorDesc preDesc    = dec.desc(embeds);
            preDesc.name          = dec.desc(embeds).name + ".pre_image_splice";
            preDesc.isInput       = false;
            preDesc.isOutput      = false;
            preDesc.isInitializer = false;
            const TensorId preEmbeds = dec.addTensor(preDesc);

            size_t producerIndex = (size_t) -1;
            for (size_t i = 0; i < dec.nodes.size(); ++i)
            {
                for (TensorId &out: dec.nodes[i].outputs)
                {
                    if (out == embeds)
                    {
                        out           = preEmbeds;
                        producerIndex = i;
                    }
                }
            }
            if (producerIndex == (size_t) -1)
            {
                return; // `embeds` is not produced internally; nothing to splice
            }

            // New graph input 1: the image feature rows, shaped exactly like the vision graph's output so
            // the caller binds that output straight across.
            TensorDesc featDesc;
            featDesc.name          = featureName;
            featDesc.shape         = featureShape;
            featDesc.dtype         = featureDtype;
            featDesc.isInput       = true;
            const TensorId features = dec.addTensor(featDesc);
            dec.inputs.push_back(features);

            // New graph input 2: for each feature row, the (batch, sequence) index it overwrites. Shape
            // [1, K, 2] matches ScatterND's index convention (the last axis addresses the first two axes of
            // the embeddings). Carried as float: the GPU ScatterND reads a float index activation and
            // truncates to int, and the caller fills in the image-token row positions (small, exact).
            const int64_t imageRows = featureShape[1];
            TensorDesc     posDesc;
            posDesc.name    = "image_positions";
            posDesc.shape   = {1, imageRows, 2};
            posDesc.dtype   = DType::Float32;
            posDesc.isInput = true;
            const TensorId positions = dec.addTensor(posDesc);
            dec.inputs.push_back(positions);

            // out = copy(preEmbeds) with the image-token rows replaced by the feature rows.
            Node scatter;
            scatter.type    = OpType::ScatterND;
            scatter.name    = "image_splice/ScatterND";
            scatter.inputs  = {preEmbeds, positions, features};
            scatter.outputs = {embeds};
            dec.nodes.insert(dec.nodes.begin() + (long) producerIndex + 1, scatter);
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
                    VKNN_INFO << "fuse: merged bucket " << a << " into bucket " << b << " on hand-off '" << boundary << "'";
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

        // Second hand-off class: the vision-language image-feature splice (see spliceImageFeatures). It is
        // a scatter, not a name match, so it is wired after the name-hand-off merges above. A decoder that
        // now produces `inputs_embeds` internally (one the embedding lookup was just merged into) and whose
        // sequence is long enough to hold the image tokens gets an ADDITIONAL, image-capable copy; the
        // original stays as the text-only decoder. The two are told apart at run time by which inputs the
        // caller binds — only an image prompt binds the feature and position inputs — so a text-only prompt
        // dispatches to the plain copy and is byte-identical to the merge, paying nothing for the splice.
        for (size_t d = 0; d < buckets.size(); ++d)
        {
            Graph         &decoder = buckets[d];
            const TensorId embeds  = decoder.find("inputs_embeds");
            if (embeds == kNoTensor || inList(decoder.inputs, embeds))
            {
                continue; // not a fused decoder: no internal token-embedding tensor to splice into
            }
            const Shape &embedsShape = decoder.desc(embeds).shape;
            if (embedsShape.size() != 3)
            {
                continue;
            }
            const int64_t seqLen = embedsShape[1];
            const int64_t hidden = embedsShape[2];

            // Find the image encoder's feature output: another bucket's graph OUTPUT shaped [1, K, hidden]
            // with K <= seqLen that no bucket consumes as an input. An unconsumed cross-graph output of the
            // right width is exactly the encoder feature the host used to splice in by hand.
            std::string featureName;
            Shape       featureShape;
            DType       featureDtype = DType::Float32;
            bool        found        = false;
            for (size_t v = 0; v < buckets.size() && !found; ++v)
            {
                if (v == d)
                {
                    continue;
                }
                for (const TensorId candidate: buckets[v].outputs)
                {
                    const TensorDesc &cd = buckets[v].desc(candidate);
                    if (cd.shape.size() != 3 || cd.shape[0] != 1 || cd.shape[2] != hidden || cd.shape[1] > seqLen)
                    {
                        continue;
                    }
                    bool consumedElsewhere = false;
                    for (size_t c = 0; c < buckets.size() && !consumedElsewhere; ++c)
                    {
                        if (c == v)
                        {
                            continue;
                        }
                        const TensorId cid = buckets[c].find(cd.name);
                        if (cid != kNoTensor && inList(buckets[c].inputs, cid))
                        {
                            consumedElsewhere = true;
                        }
                    }
                    if (consumedElsewhere)
                    {
                        continue;
                    }
                    featureName  = cd.name;
                    featureShape = cd.shape;
                    featureDtype = cd.dtype;
                    found        = true;
                    break;
                }
            }
            if (!found)
            {
                continue;
            }

            // Keep `decoder` as the text-only path; add an image-capable copy right after it.
            Graph imageDecoder = decoder;
            spliceImageFeatures(imageDecoder, imageDecoder.find("inputs_embeds"), featureName, featureShape, featureDtype);
            VKNN_INFO << "fuse: added on-GPU image-feature splice (ScatterND on '" << featureName << "') as an image copy of bucket " << d;
            buckets.insert(buckets.begin() + (long) d + 1, std::move(imageDecoder));
            names.insert(names.begin() + (long) d + 1, names[d] + " (image)");
            ++d; // skip the copy just inserted
        }
    }

} // namespace vknn
