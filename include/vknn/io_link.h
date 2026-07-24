// Caller-declared linkage between a graph output and a graph input for engine-resident I/O.
#pragma once
#include <cstdint>
#include <vector>

namespace vknn {

    /// One contiguous element range copied from a linked output into a linked input at the start of
    /// a run (see Session::linkOutputToInput). Offsets and count are CANONICAL elements — positions
    /// in the tensor's logical NCHW row-major order — independent of the device layout or storage
    /// precision; the engine maps them to the backing layout. A recurrent-state caller (e.g. an
    /// autoregressive decoder's KV cache) updates these per run to place the newly produced rows.
    struct LinkRange {
        int64_t sourceElem = 0; ///< First element read from the linked output.
        int64_t destElem   = 0; ///< First element written in the linked input.
        int64_t count      = 0; ///< Number of elements copied.
    };

    /// Chunk length of the auto-emitted chunk-prefill bucket (vknn_compile) and the prompt window
    /// the chunked prefill driver feeds per pass. 64 balances the two costs of a fixed-size chunk:
    /// a larger chunk amortizes per-pass overhead (dispatch, mask/position pack, one submit per
    /// chunk) over more prompt tokens, while every prompt pays the FULL chunk's compute for its
    /// last, padded chunk — at 64 the average last-chunk waste is 32 token-columns, small against
    /// the per-pass fixed cost it removes. One static [1, 64] graph serves every prompt length as
    /// ceil(T / 64) sequential passes over the cached KV (the llm.npu chunked-prefill scheme,
    /// arXiv 2407.05858), so a variable-length prompt never needs a per-length plan or a
    /// pad-to-full-window forward.
    inline constexpr int64_t kChunkPrefillTokens = 64;

    /// The cache slot a decode step at absolute position `position` folds the PREVIOUS token's
    /// present row into: slot position-1, clamped to the last slot once the position runs past the
    /// compiled context window (the overrun then keeps overwriting the newest slot). Negative (no
    /// pending fold) at position 0. The single source of the fold-slot rule shared by the
    /// single-step decode loop and the per-iteration precompute of a decode chain.
    inline int64_t kvFoldSlot(int64_t position, int64_t cacheSlots) {
        if (position <= 0)
        {
            return -1;
        }
        const int64_t slot = position - 1;
        return slot < cacheSlots ? slot : cacheSlots - 1;
    }

    /// Ranges for a row-block KV fold: copy `rows` consecutive rows starting at row `sourceRow` of
    /// a linked present output [1, kvHeads, presentRows, headDim] into slots `slot`..`slot`+rows-1
    /// of a linked past input [1, kvHeads, cacheSlots, headDim], one contiguous range per head
    /// (rows of one head are contiguous in row-major storage). `slot` < 0 or `rows` <= 0 returns no
    /// ranges (the reset / first-step state with no pending fold). The caller keeps the block in
    /// bounds on both sides; the link validation rejects an overrun. A chunked prefill folds each
    /// chunk's `rows` produced KV rows into their absolute cache slots with this; the single-token
    /// decode fold is the rows == 1 case below.
    inline std::vector<LinkRange> kvFoldRowRanges(int64_t kvHeads, int64_t presentRows, int64_t cacheSlots, int64_t headDim, int64_t sourceRow, int64_t slot, int64_t rows) {
        std::vector<LinkRange> ranges;
        if (slot < 0 || rows <= 0)
        {
            return ranges;
        }
        ranges.reserve((size_t) kvHeads);
        for (int64_t head = 0; head < kvHeads; ++head)
        {
            ranges.push_back({(head * presentRows + sourceRow) * headDim, (head * cacheSlots + slot) * headDim, rows * headDim});
        }
        return ranges;
    }

    /// Ranges for a single-token KV fold: copy the NEWEST row of a linked present output
    /// [1, kvHeads, presentRows, headDim] into slot `slot` of a linked past input
    /// [1, kvHeads, cacheSlots, headDim], one range per head. The newest row is the LAST present
    /// row, which covers both with-past decoder conventions: a present that concatenates the cache
    /// with the new token (presentRows == cacheSlots + 1; the new row sits at index cacheSlots) and
    /// a present that carries only the produced rows (presentRows == 1 for a one-token step; the new
    /// row is row 0). `presentRows` MUST come from the decode plan's present output shape — assuming
    /// one convention breaks the other (a cache-concat offset applied to a rows-only present is out
    /// of bounds and rejected by the link validation). `slot` < 0 returns no ranges (the reset /
    /// first-step state with no pending fold).
    inline std::vector<LinkRange> kvFoldRanges(int64_t kvHeads, int64_t presentRows, int64_t cacheSlots, int64_t headDim, int64_t slot) {
        return kvFoldRowRanges(kvHeads, presentRows, cacheSlots, headDim, presentRows - 1, slot, 1);
    }

} // namespace vknn
