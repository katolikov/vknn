// Greedy speculative decoding: the rules a driver applies to turn a small draft decoder's guesses
// into the target decoder's own greedy token stream, paid for with ONE batched target forward per
// round instead of one forward per token.
//
// WHY IT PAYS HERE. Decode on this engine is bandwidth-bound: a single-token step is a split-K
// GEMV that reads every weight byte to produce one token, so a 0.5B int4 decoder streams its whole
// ~520 MB weight set per token and the arithmetic units idle. A verification pass over k+1 token
// columns reads the SAME weight bytes once and produces up to k+1 committed tokens, turning the
// GEMV into a small-batch GEMM with k+1x the arithmetic intensity. The draft decoder pays its own
// (much smaller) weight traffic k times per round. Weight traffic per accepted token is
//
//     (W_target + k * W_draft) / (1 + E[accepted])
//
// against W_target for plain decode, so the win is set by the acceptance rate and the draft/target
// size ratio, not by any kernel change. (Mechanism reference: Lever, arXiv 2605.16786, which
// applies it to flash-resident NPU decode; the scheme itself is backend-agnostic.)
//
// EXACTNESS. This is the greedy variant, which is exactly equivalent BY CONSTRUCTION rather than
// approximately: a drafted token is accepted only when it EQUALS the target's own argmax at that
// position, and the first mismatch is replaced by the target's argmax and truncates the round. Every
// emitted token is therefore an argmax of a target forward over exactly the prefix the plain decode
// loop would have fed. The token stream is not "close to" the non-speculative stream; it is that
// stream. Any divergence is a bug.
//
// The one assumption behind that statement is that the target's logits for a given prefix do not
// depend on how many token columns share the forward — the batched verification bucket (M = k+1)
// must agree with the decode bucket (M == 1) to the precision that decides an argmax. That is the
// same property the chunked-prefill path already rests on (io_link.h), and it is a DEVICE property:
// the host CPU-backend gate proves the rule, the device gate proves the kernels agree.
//
// SAMPLING IS OUT OF SCOPE. Temperature / top-k / top-p speculation needs the modified-rejection
// scheme (accept with probability min(1, p_target/p_draft), else resample from the normalized
// positive part of p_target - p_draft), which requires the full target AND draft distributions per
// position and a random draw per token. v1 is greedy-only and refuses to speculate under a nonzero
// temperature, so nothing in this header depends on an RNG or on a run's timing.
//
// ---- the round --------------------------------------------------------------------------------
//
// State entering a round, matching the plain decode loop's: the KV cache holds rows for absolute
// positions 0..p-1, and `next` is the token at position p, produced by the previous forward and not
// yet fed.
//
//  1. DRAFT. The draft decoder feeds `next` at p and then its own output at p+1, p+2, ..., k
//     forwards in all, yielding proposals d[0..k) for positions p+1..p+k.
//  2. VERIFY. The target runs ONE forward over the k+1 token columns [next, d[0], .., d[k-1]] at
//     positions p..p+k, through a bucket compiled at input_ids [1, kSpecVerifyTokens]. Row i of its
//     logits is the target's prediction for position p+1+i, so argmax row i is a[i], i in [0, k].
//  3. ACCEPT. j = specAcceptedDrafts(d, a, k) — the longest prefix with d[i] == a[i]. The round
//     commits the token list [next, d[0], .., d[j-1]] (1 + j tokens, positions p..p+j) and leaves
//     a[j] as the new `next` at position p+1+j: a[j] is the CORRECTION when j < k and the free BONUS
//     token when j == k. a[j] always exists, which is why the pass verifies k+1 columns for k drafts.
//
// Every committed token is the argmax of a target row conditioned on the committed prefix before it,
// which is the plain loop's definition of the next token, so the streams coincide. A draft that is
// always wrong commits exactly one token per round (j == 0) — the plain stream at extra cost, never
// a different stream.
//
// ---- KV cache rollback ------------------------------------------------------------------------
//
// The verification forward produces present rows for ALL k+1 fed columns, of which only the first
// 1 + j belong to the conversation. VKNN never writes present rows into cache slots as a side effect
// of a run: a linked present->past pair copies exactly the LinkRange list the caller arms before the
// NEXT run (io_link.h). Rollback is therefore not an erase — it is a shorter range list.
// specVerifyFoldRanges() folds rows 0..j of the produced block into slots p..p+j and names no source
// row past that, so a rejected row is never copied anywhere and the cache after a partially accepted
// round is byte-identical to the cache the plain loop would hold. Nothing has to be undone, which is
// what makes the rollback trivially correct rather than a state-machine hazard.
//
// The DRAFT model's cache is a different matter and deliberately a weaker one. Its rows for rejected
// positions are stale, but a decoder only attends cache slots the attention mask marks valid, and the
// mask is derived from the round's committed position p — so slots >= p are invisible and are
// overwritten by the next round's drafting before they ever become visible. More fundamentally: the
// draft model is outside the correctness argument entirely. Its weights, its precision, its cache,
// even a completely wrong rollback can only change WHICH tokens get proposed, and a proposal is only
// ever emitted after the target's own argmax has confirmed it. A draft bug costs throughput, never
// tokens.
//
// ---- degradation ------------------------------------------------------------------------------
//
// Speculation is an optimization with a prerequisite the engine cannot manufacture — a second model.
// Every way that prerequisite can be missing degrades to the plain decode loop with one notice and
// the identical stream: no draft artifact supplied, a draft that fails to load, a draft whose vocab
// differs from the target's (its ids would not be the target's ids), a target .vxm with no
// kSpecVerifyTokens bucket (compiled before the bucket existed, or a decoder the plan refuses), a
// nonzero temperature, and a position too close to the compiled context edge to fit a whole
// verification window. None of these is an error.
#pragma once
#include "vknn/io_link.h"
#include <cstdint>
#include <vector>

namespace vknn {

    /// Tokens the draft decoder proposes per speculative round, and so the number of draft forwards
    /// per round. 4 sits where the two costs of a fixed k cross: weight traffic per accepted token
    /// falls as 1/(1 + E[accepted]) and is still improving at 4, while the drafts are speculative
    /// work that is thrown away on rejection — at a typical 0.6-0.8 per-token acceptance for a
    /// same-family draft, the expected run length before a mismatch is ~2.5-4 tokens, so proposals
    /// past the fourth are rejected more often than not and their draft forwards are pure loss. It
    /// also keeps the verification window narrow enough that the M > 1 attention path stays cheap
    /// against the weight read it shares. Compiled into the .vxm as the verification bucket's width,
    /// so it is a build constant, not a runtime knob.
    inline constexpr int64_t kSpecDraftTokens = 4;

    /// Token columns the verification forward covers: the k drafts plus the one already-committed
    /// token they extend. The extra column is what makes the round self-correcting — its logits row
    /// gives the target's own next token, which replaces a rejected draft (and, when every draft is
    /// accepted, is emitted as a free bonus token).
    inline constexpr int64_t kSpecVerifyTokens = kSpecDraftTokens + 1;

    /// Whether vknn_compile emits the verification bucket at all. The bucket costs graph metadata
    /// only (it shares the deduped initializer pool with the decode bucket), and a driver with no
    /// draft model never dispatches to it, so emitting it unconditionally keeps every .vxm
    /// speculation-ready without changing any existing path.
    inline constexpr bool kSpecDecodeEnabled = true;

    /// The longest prefix of `drafts` the target confirms: the largest j <= draftCount with
    /// drafts[i] == targetArgMax[i] for every i < j. `targetArgMax` holds draftCount + 1 entries —
    /// row i is the target's own token for the position drafts[i] occupies, and the last entry is
    /// the token after the final draft. The acceptance test is exact integer equality on token ids;
    /// there is no threshold, no probability, and no dependence on anything but the two id lists.
    inline int specAcceptedDrafts(const int64_t *drafts, const int64_t *targetArgMax, int draftCount) noexcept {
        int accepted = 0;
        while (accepted < draftCount && drafts[accepted] == targetArgMax[accepted])
        {
            ++accepted;
        }
        return accepted;
    }

    /// How many of a round's committed tokens the turn actually emits and feeds, given the plain
    /// decode loop's two stopping rules. `committed` is [next, drafts[0..accepted)] in order,
    /// `count` its length. The plain loop stops at the FIRST eos without printing or feeding it, and
    /// stops after `budget` more tokens; both truncations must apply here too, or a speculative turn
    /// would emit past the point the plain turn ends and fold cache rows the plain turn never wrote.
    /// A negative `eos` never matches (the disable-early-stop sentinel).
    inline int specEmittedCount(const int64_t *committed, int count, int64_t eos, int budget) noexcept {
        int emitted = count < budget ? count : budget;
        for (int i = 0; i < emitted; ++i)
        {
            if (committed[i] == eos)
            {
                return i;
            }
        }
        return emitted;
    }

    /// Ranges that fold a verification pass's ACCEPTED present rows into the cache, and only those:
    /// `rows` consecutive rows starting at the first produced row of a
    /// [1, kvHeads, presentRows, headDim] present output, into slots `slot`..`slot`+rows-1 of the
    /// [1, kvHeads, cacheSlots, headDim] past input. `verifyTokens` is the bucket's window width, so
    /// the produced block starts at presentRows - verifyTokens — which covers both present
    /// conventions (a cache-concat present carries the cache rows first, a rows-only present starts
    /// at 0). Rows past `slot`+rows-1 are the rejected drafts and appear in no range, so they are
    /// never copied into the cache. `rows` <= 0 arms an empty list (a round that committed nothing,
    /// and the first pass of a turn, which re-seeds the cache by binding it instead).
    inline std::vector<LinkRange> specVerifyFoldRanges(int64_t kvHeads, int64_t presentRows, int64_t cacheSlots, int64_t headDim, int64_t verifyTokens, int64_t slot, int64_t rows) {
        return kvFoldRowRanges(kvHeads, presentRows, cacheSlots, headDim, presentRows - verifyTokens, rows > 0 ? slot : -1, rows);
    }

} // namespace vknn
