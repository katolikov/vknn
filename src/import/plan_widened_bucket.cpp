// planWidenedDecodeBucket: derive the shape set for an automatic bucket that runs one with-past
// decode graph at several token columns per step.
//
// A decoder's decode bucket fixes the whole cache geometry: input_ids [1,1],
// past_key_values.*.{key,value} [1, kvHeads, C, headDim], an attention mask spanning the C past
// slots plus the step's tokens, and (for the drivers' batched conventions) position_ids. A widened
// bucket is the SAME graph compiled at `width` tokens per step: ids and positions become
// [1, width], the mask gains width - 1 columns, and every past input keeps its decode shape — so a
// batched pass over `width` tokens runs against the cache the decode bucket already owns, under one
// static plan.
//
// Two roles are built on it, differing only in width and in their duplicate policy:
// chunk-prefill at kChunkPrefillTokens (plan_chunk_prefill.cpp) and speculative verification at
// kSpecVerifyTokens (plan_spec_verify.cpp).
//
// This pass only PLANS the bucket (the input-shape map a fresh import compiles under); vknn_compile
// performs the import and appends the result to the bucket list, so existing bucket indices and old
// readers are unaffected.
#include "passes_internal.h"

namespace vknn {

    TensorId graphInputByName(const Graph &g, const std::string &name) {
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

    namespace {

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

    bool planWidenedDecodeBucket(const std::vector<Graph> &buckets, int64_t width, WidenedDecodePlan *plan) {
        if (width < 2)
        {
            return false; // a width-1 bucket IS the decode bucket
        }
        for (size_t b = 0; b < buckets.size(); ++b)
        {
            const Graph   &g   = buckets[b];
            const TensorId ids = graphInputByName(g, "input_ids");
            if (ids == kNoTensor || g.desc(ids).shape != Shape {1, 1})
            {
                continue;
            }
            const TensorId pos  = graphInputByName(g, "position_ids");
            const TensorId mask = graphInputByName(g, "attention_mask");
            const TensorId pk0  = graphInputByName(g, "past_key_values.0.key");
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
            if (cacheSlots < width)
            {
                continue; // the runtime gates a pass at p + width <= C; a full window could never run
            }
            if (!hasOutput(g, "logits") || !hasOutput(g, "present.0.key"))
            {
                continue;
            }
            // The mask widens from the decode step's 1 token to the window's tokens: the 2-D token
            // mask [1, C+1] -> [1, C+width]; the 4-D additive mask [1, x, 1, C+1] ->
            // [1, x, width, C+width] (axis 2 is the query axis, axis 3 the key columns).
            const Shape &maskShape = g.desc(mask).shape;
            Shape        wideMask;
            if (maskShape.size() == 2 && maskShape[1] == cacheSlots + 1)
            {
                wideMask = {maskShape[0], cacheSlots + width};
            } else if (maskShape.size() == 4 && maskShape[2] == 1 && maskShape[3] == cacheSlots + 1)
            {
                wideMask = {maskShape[0], maskShape[1], width, cacheSlots + width};
            } else
            {
                continue;
            }
            plan->decodeBucket = b;
            plan->cacheSlots   = cacheSlots;
            plan->label.clear();
            plan->shapes.clear();
            for (TensorId in: g.inputs)
            {
                const TensorDesc &d = g.desc(in);
                if (in == ids || in == pos)
                {
                    plan->shapes[d.name] = {1, width};
                } else if (in == mask)
                {
                    plan->shapes[d.name] = wideMask;
                } else
                {
                    // Past key/value inputs and any extra window-independent input keep the decode
                    // bucket's shape verbatim; a window-dependent extra input surfaces as a shape
                    // conflict in the widened bucket's own compile, which the caller reports and
                    // skips rather than emitting a broken bucket.
                    plan->shapes[d.name] = d.shape;
                }
            }
            return true;
        }
        return false;
    }

} // namespace vknn
