// Static, deterministic loop partitioning for the CPU backend, backed by a persistent thread pool.
#pragma once
#include "vknn/config_struct.h"
#include <algorithm>
#include <cstdint>
#include <functional>

namespace vknn { namespace cpu {

    namespace detail {
        /// Run `task(i)` for i in [0, nTasks). Task 0 runs on the calling thread; tasks 1..n-1 run
        /// on persistent pool workers. Blocks until every task has finished. The first exception a
        /// task throws is rethrown on the caller after the join, so an operator's Error still
        /// propagates out of run().
        void dispatch(int nTasks, const std::function<void(int)> &task);
        /// True while the calling thread is executing a dispatch() task. A nested parallelFor runs
        /// serially rather than recursing into the pool.
        bool inParallelRegion() noexcept;
        /// Worker threads the pool has spun up (excluding the caller). Test/diagnostic accessor.
        int poolWorkers() noexcept;
        /// Parallel regions the pool has run since process start. Test/diagnostic accessor; a test can
        /// assert it advanced to prove a parallelFor actually partitioned rather than running inline.
        int64_t poolDispatches() noexcept;
    } // namespace detail

    /// Worker count for the run `c` configures: Config::cpuThreads clamped to at least 1, and 1
    /// when no config is bound to the ExecContext.
    inline int threadCount(const Config *c) noexcept {
        return c ? std::max(1, c->cpuThreads) : 1;
    }

    /// Scalar operations a chunk must be worth before handing it to a thread. A pool dispatch costs a
    /// few microseconds; at roughly one scalar op per nanosecond this covers it with margin.
    inline constexpr int64_t kMinChunkOps = 16384;

    /// parallelFor's `minChunk` for a loop whose one iteration costs about `opsPerIteration` scalar
    /// operations — an elementwise sweep passes 1, a conv output plane passes its tap count. A loop
    /// too cheap overall to reach kMinChunkOps in any chunk ends up running inline.
    inline int64_t minChunkForWork(int64_t opsPerIteration) noexcept {
        int64_t ops = opsPerIteration > 0 ? opsPerIteration : 1;
        return ops >= kMinChunkOps ? 1 : (kMinChunkOps + ops - 1) / ops;
    }

    /// Split [begin, end) into contiguous chunks and invoke `body(chunkBegin, chunkEnd)` on each, in
    /// parallel. The chunk count is `min(threads, (end-begin)/minChunk)`; chunk i spans `base =
    /// (end-begin)/t` iterations plus one more for the first `(end-begin) % t` chunks. Boundaries
    /// therefore depend only on (begin, end, t), never on scheduling.
    ///
    /// `minChunk` is the smallest iteration count worth a thread: a range too short to give every
    /// thread that many runs on fewer threads, or inline.
    ///
    /// The caller is responsible for the only correctness precondition: iterations of [begin, end)
    /// must write DISJOINT outputs and carry no cross-iteration accumulation. Under that condition
    /// each output element is computed by exactly the same expression, in the same order, as the
    /// serial loop — so the result is bit-identical to `threads == 1` and the CPU backend keeps its
    /// role as the byte oracle. Partitioning a REDUCTION loop would reassociate its sum and is
    /// never valid here.
    ///
    /// `threads <= 1`, a short range, or a nested call runs `body(begin, end)` inline on the caller
    /// with no pool involvement.
    template <class Fn> void parallelFor(int threads, int64_t begin, int64_t end, int64_t minChunk, Fn &&body) {
        int64_t n = end - begin;
        if (n <= 0)
        {
            return;
        }
        int64_t cap = n / (minChunk > 0 ? minChunk : 1);
        int     t   = threads;
        if ((int64_t) t > cap)
        {
            t = (int) cap;
        }
        if (t <= 1 || detail::inParallelRegion())
        {
            body(begin, end);
            return;
        }
        const int64_t base = n / t;
        const int64_t rem  = n % t;
        detail::dispatch(t, [&](int i) {
            int64_t b = begin + i * base + std::min<int64_t>(i, rem);
            int64_t e = b + base + (i < rem ? 1 : 0);
            if (e > b)
            {
                body(b, e);
            }
        });
    }

}} // namespace vknn::cpu
