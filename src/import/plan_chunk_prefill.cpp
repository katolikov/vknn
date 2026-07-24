// planChunkPrefillBucket: derive the shape set for the automatic chunk-prefill bucket of a
// multi-bucket LLM compile.
//
// A with-past decoder's decode bucket fixes the whole cache geometry: input_ids [1,1],
// past_key_values.*.{key,value} [1, kvHeads, C, headDim], an attention mask spanning the C past
// slots plus the step's tokens, and (for the drivers' batched-prefill convention) position_ids.
// The chunk-prefill bucket is the SAME graph compiled at kChunkPrefillTokens tokens per step
// (io_link.h documents the chunk size): input_ids/position_ids widen to [1, chunk], the mask to
// past + chunk columns, and every past input keeps its decode shape — so ceil(T / chunk)
// sequential passes prefill any prompt length T over the cached KV against ONE static plan
// (the llm.npu chunked-prefill scheme, arXiv 2407.05858). This pass only PLANS the bucket (the
// input-shape map a fresh import compiles under); vknn_compile performs the import and appends
// the result to the bucket list, so existing bucket indices and old readers are unaffected.
//
// Refusals return false and leave the compile without a chunk bucket: no qualifying decode
// bucket, a compiled context shorter than one chunk (the runtime could never run a full chunk),
// a mask outside the two known conventions (2-D [1, C+1] token mask or 4-D [1, x, 1, C+1]
// additive mask), a missing position_ids input (the chunked driver feeds absolute positions
// explicitly; a mask-derived-position export cannot take padded chunks), or an already
// chunk-capable prefill bucket.
#include "import/passes.h"
#include "vknn/io_link.h"

namespace vknn {

    namespace {

        // The graph-input tensor named `name`, or kNoTensor when absent or not a graph input.
        TensorId inputByName(const Graph &g, const std::string &name) {
            const TensorId id = g.find(name);
            if (id == kNoTensor)
            {
                return kNoTensor;
            }
            for (TensorId in: g.inputs)
            {
                if (in == id)
                {
                    return id;
                }
            }
            return kNoTensor;
        }

        bool hasOutput(const Graph &g, const std::string &name) {
            const TensorId id = g.find(name);
            if (id == kNoTensor)
            {
                return false;
            }
            for (TensorId out: g.outputs)
            {
                if (out == id)
                {
                    return true;
                }
            }
            return false;
        }

    } // namespace

    bool planChunkPrefillBucket(const std::vector<Graph> &buckets, ChunkPrefillPlan *plan) {
        const int64_t chunk = kChunkPrefillTokens;
        for (size_t b = 0; b < buckets.size(); ++b)
        {
            const Graph   &g   = buckets[b];
            const TensorId ids = inputByName(g, "input_ids");
            if (ids == kNoTensor || g.desc(ids).shape != Shape {1, 1})
            {
                continue;
            }
            const TensorId pos  = inputByName(g, "position_ids");
            const TensorId mask = inputByName(g, "attention_mask");
            const TensorId pk0  = inputByName(g, "past_key_values.0.key");
            if (pos == kNoTensor || g.desc(pos).shape != Shape {1, 1} || mask == kNoTensor || pk0 == kNoTensor)
            {
                continue;
            }
            const Shape &cacheShape = g.desc(pk0).shape; // [1, kvHeads, C, headDim]
            if (cacheShape.size() != 4 || cacheShape[0] != 1)
            {
                continue;
            }
            const int64_t cacheSlots = cacheShape[2];
            if (cacheSlots < chunk)
            {
                continue; // the runtime gates chunks at p + chunk <= C; a full chunk could never run
            }
            if (!hasOutput(g, "logits") || !hasOutput(g, "present.0.key"))
            {
                continue;
            }
            // The mask widens from the decode step's 1 token to the chunk's tokens: the 2-D token
            // mask [1, C+1] -> [1, C+chunk]; the 4-D additive mask [1, x, 1, C+1] ->
            // [1, x, chunk, C+chunk] (axis 2 is the query axis, axis 3 the key columns).
            const Shape &maskShape = g.desc(mask).shape;
            Shape        chunkMask;
            if (maskShape.size() == 2 && maskShape[1] == cacheSlots + 1)
            {
                chunkMask = {maskShape[0], cacheSlots + chunk};
            } else if (maskShape.size() == 4 && maskShape[2] == 1 && maskShape[3] == cacheSlots + 1)
            {
                chunkMask = {maskShape[0], maskShape[1], chunk, cacheSlots + chunk};
            } else
            {
                continue;
            }
            // An existing prefill bucket at S <= chunk over the same cache already serves the
            // chunked driver; a second one would only duplicate the plan.
            bool alreadyChunkCapable = false;
            for (const Graph &other: buckets)
            {
                const TensorId oIds = inputByName(other, "input_ids");
                const TensorId oPk  = inputByName(other, "past_key_values.0.key");
                if (oIds == kNoTensor || oPk == kNoTensor)
                {
                    continue;
                }
                const Shape &oShape = other.desc(oIds).shape;
                alreadyChunkCapable = alreadyChunkCapable || (oShape.size() == 2 && oShape[0] == 1 && oShape[1] > 1 && oShape[1] <= chunk && other.desc(oPk).shape == cacheShape);
            }
            if (alreadyChunkCapable)
            {
                return false;
            }
            plan->decodeBucket = b;
            plan->shapes.clear();
            for (TensorId in: g.inputs)
            {
                const TensorDesc &d = g.desc(in);
                if (in == ids || in == pos)
                {
                    plan->shapes[d.name] = {1, chunk};
                } else if (in == mask)
                {
                    plan->shapes[d.name] = chunkMask;
                } else
                {
                    // Past key/value inputs and any extra S-independent input keep the decode
                    // bucket's shape verbatim; an S-dependent extra input surfaces as a shape
                    // conflict in the chunk bucket's own compile, which the caller reports and
                    // skips rather than emitting a broken bucket.
                    plan->shapes[d.name] = d.shape;
                }
            }
            plan->label = "chunk-prefill:input_ids=1x" + std::to_string(chunk);
            return true;
        }
        return false;
    }

} // namespace vknn
