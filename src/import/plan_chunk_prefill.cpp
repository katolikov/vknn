// planChunkPrefillBucket: the chunk-prefill role of the widened-decode bucket plan.
//
// The chunk-prefill bucket is a with-past decode graph compiled at kChunkPrefillTokens tokens per
// step (io_link.h documents the chunk size), so ceil(T / chunk) sequential passes prefill any prompt
// length T over the cached KV against ONE static plan (the llm.npu chunked-prefill scheme,
// arXiv 2407.05858). planWidenedDecodeBucket derives the geometry; this file adds the role's name
// and its duplicate policy.
//
// Refusals return false and leave the compile without a chunk bucket: everything
// planWidenedDecodeBucket refuses (no qualifying decode bucket, a compiled context shorter than one
// chunk, a mask outside the two known conventions, a missing position_ids input — the chunked driver
// feeds absolute positions explicitly, so a mask-derived-position export cannot take padded chunks),
// plus an already chunk-capable prefill bucket, which would only duplicate the plan.
#include "passes_internal.h"
#include "vknn/io_link.h"

namespace vknn {

    bool planChunkPrefillBucket(const std::vector<Graph> &buckets, ChunkPrefillPlan *plan) {
        const int64_t chunk = kChunkPrefillTokens;
        if (!planWidenedDecodeBucket(buckets, chunk, plan))
        {
            return false;
        }
        // An existing prefill bucket at S <= chunk over the same cache already serves the chunked
        // driver. The cache shape comes from the planned decode bucket, so a bucket over a different
        // cache geometry (a second model in a multi-graph compile) does not count.
        const Shape &cacheShape = plan->shapes.at("past_key_values.0.key");
        for (const Graph &other: buckets)
        {
            const TensorId oIds = graphInputByName(other, "input_ids");
            const TensorId oPk  = graphInputByName(other, "past_key_values.0.key");
            if (oIds == kNoTensor || oPk == kNoTensor)
            {
                continue;
            }
            const Shape &oShape = other.desc(oIds).shape;
            if (oShape.size() == 2 && oShape[0] == 1 && oShape[1] > 1 && oShape[1] <= chunk && other.desc(oPk).shape == cacheShape)
            {
                return false;
            }
        }
        plan->label = "chunk-prefill:input_ids=1x" + std::to_string(chunk);
        return true;
    }

} // namespace vknn
