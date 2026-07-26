// planSpecVerifyBucket: the speculative-verification role of the widened-decode bucket plan.
//
// The verification bucket is a with-past decode graph compiled at kSpecVerifyTokens tokens per step
// (spec_decode.h): one forward checks kSpecDraftTokens proposals from a small draft decoder against
// the target's own argmax, plus the anchor token they extend, so a greedy speculative round pays the
// target's weight traffic ONCE for up to kSpecDraftTokens + 1 committed tokens. planWidenedDecodeBucket
// derives the geometry; this file adds the role's name and its duplicate policy.
//
// The bucket is emitted whether or not a draft model ever exists: it shares the compile's deduped
// initializer pool, so it costs graph metadata rather than weights, and a driver with no draft never
// dispatches to it. That keeps a .vxm speculation-ready without a compile-time decision the caller
// would have to have made correctly months earlier.
//
// Refusals return false and leave the compile without a verification bucket (the driver then decodes
// token by token, same stream): everything planWidenedDecodeBucket refuses — no qualifying decode
// bucket, a compiled context shorter than one verification window, a mask outside the two known
// conventions, a missing position_ids input (the verification pass feeds the window's absolute
// positions explicitly, so a mask-derived-position export cannot take it) — plus a bucket already
// compiled at exactly this width, which already serves the pass.
#include "passes_internal.h"
#include "vknn/spec_decode.h"

namespace vknn {

    bool planSpecVerifyBucket(const std::vector<Graph> &buckets, SpecVerifyPlan *plan) {
        if (!kSpecDecodeEnabled)
        {
            return false;
        }
        const int64_t verify = kSpecVerifyTokens;
        if (!planWidenedDecodeBucket(buckets, verify, plan))
        {
            return false;
        }
        // A bucket already at exactly this width over the same cache IS the verification bucket (a
        // recompile of a .vxm that carries one, or a hand-declared --bucket at the same shape).
        // Unlike the chunk-prefill policy this refuses only on an exact width match: a wider prefill
        // bucket cannot verify a kSpecVerifyTokens window without padding the round out to its own
        // width, which would pay that window's attention cost every round.
        const Shape &cacheShape = plan->shapes.at("past_key_values.0.key");
        for (const Graph &other: buckets)
        {
            const TensorId oIds = graphInputByName(other, "input_ids");
            const TensorId oPk  = graphInputByName(other, "past_key_values.0.key");
            if (oIds == kNoTensor || oPk == kNoTensor)
            {
                continue;
            }
            if (other.desc(oIds).shape == Shape {1, verify} && other.desc(oPk).shape == cacheShape)
            {
                return false;
            }
        }
        plan->label = "spec-verify:input_ids=1x" + std::to_string(verify);
        return true;
    }

} // namespace vknn
