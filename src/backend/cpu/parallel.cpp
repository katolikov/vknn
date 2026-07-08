// Persistent worker pool behind cpu::parallelFor.
#include "backend/cpu/parallel.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace vknn { namespace cpu { namespace detail {

    namespace {

        /// Set for the duration of a dispatch() task, on the caller as well as on the workers, so a
        /// nested parallelFor detects the region and runs its body inline instead of recursing into
        /// the pool.
        thread_local bool tlInParallel = false;

        /// One process-wide pool of workers, shared by every CPU segment.
        ///
        /// Sizing: the pool grows to the largest task count any dispatch has asked for and never
        /// shrinks, so a steady-state run spawns no threads.
        ///
        /// Lifecycle of one dispatch: `dispatchMu_` admits a single region at a time; the caller
        /// publishes the task and bumps `gen_`, waking every worker. Each worker claims indices from
        /// the atomic `next_` until they run out, then retires one unit of `awake_`. The caller runs
        /// index 0 itself and returns once `awake_` reaches zero — which is also the point at which no
        /// worker can still be reading `task_`/`nTasks_`, so the next dispatch may overwrite them.
        ///
        /// Which worker runs which index is unspecified and irrelevant: parallelFor's contract is that
        /// tasks write disjoint outputs, so the result never depends on the schedule.
        class Pool {
          public:
            static Pool &instance() {
                static Pool p;
                return p;
            }

            ~Pool() {
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    stop_ = true;
                    ++gen_;
                }
                cv_.notify_all();
                for (std::thread &t: workers_)
                {
                    if (t.joinable())
                    {
                        t.join();
                    }
                }
            }

            void dispatch(int nTasks, const std::function<void(int)> &task) {
                std::lock_guard<std::mutex> region(dispatchMu_);
                dispatches_.fetch_add(1, std::memory_order_relaxed);
                ensure(nTasks - 1);
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    task_   = &task;
                    nTasks_ = nTasks;
                    next_.store(1, std::memory_order_relaxed);
                    awake_ = (int) workers_.size(); // every worker wakes; each retires once
                    err_   = nullptr;
                    ++gen_;
                }
                cv_.notify_all();
                runTask(0);
                std::exception_ptr e;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    doneCv_.wait(lk, [this] { return awake_ == 0; });
                    e     = err_;
                    err_  = nullptr;
                    task_ = nullptr;
                }
                if (e)
                {
                    std::rethrow_exception(e);
                }
            }

            int workerCount() {
                std::lock_guard<std::mutex> lk(mu_);
                return (int) workers_.size();
            }

            int64_t dispatchCount() const noexcept {
                return dispatches_.load(std::memory_order_relaxed);
            }

          private:
            Pool() = default;

            /// Grow to at least `want` workers. Called under dispatchMu_ and before the generation
            /// bump, so a worker started here carries the pre-bump generation as its `seen` value and
            /// therefore joins the very dispatch that created it (rather than sleeping through it and
            /// stalling the caller's `awake_` countdown).
            void ensure(int want) {
                std::lock_guard<std::mutex> lk(mu_);
                while ((int) workers_.size() < want)
                {
                    workers_.emplace_back([this, g = gen_] { workerLoop(g); });
                }
            }

            /// Run task index `i`, capturing the first exception any task throws so an operator's Error
            /// propagates out of the dispatch rather than terminating the worker.
            void runTask(int i) {
                bool outer   = !tlInParallel;
                tlInParallel = true;
                try
                {
                    (*task_)(i);
                }
                catch (...)
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    if (!err_)
                    {
                        err_ = std::current_exception();
                    }
                }
                if (outer)
                {
                    tlInParallel = false;
                }
            }

            void workerLoop(uint64_t seen) {
                tlInParallel = true; // a worker only ever runs inside a dispatch region
                for (;;)
                {
                    {
                        std::unique_lock<std::mutex> lk(mu_);
                        cv_.wait(lk, [this, seen] { return stop_ || gen_ != seen; });
                        seen = gen_;
                        if (stop_)
                        {
                            return;
                        }
                    }
                    // mu_ was just released after an acquire on the publishing store of task_ / nTasks_,
                    // so both are visible and stay stable until this worker retires.
                    for (;;)
                    {
                        int i = next_.fetch_add(1, std::memory_order_relaxed);
                        if (i >= nTasks_)
                        {
                            break;
                        }
                        runTask(i);
                    }
                    bool last = false;
                    {
                        std::lock_guard<std::mutex> lk(mu_);
                        last = (--awake_ == 0);
                    }
                    if (last)
                    {
                        doneCv_.notify_all();
                    }
                }
            }

            std::mutex                      dispatchMu_; ///< admits one dispatch region at a time
            std::mutex                      mu_;         ///< guards workers_, gen_, stop_, task_, nTasks_, awake_, err_
            std::condition_variable         cv_;         ///< wakes workers on a new generation
            std::condition_variable         doneCv_;     ///< wakes the caller once awake_ hits 0
            std::vector<std::thread>        workers_;
            const std::function<void(int)> *task_   = nullptr;
            int                             nTasks_ = 0;
            std::atomic<int>                next_ {0};
            int                             awake_ = 0;
            std::exception_ptr              err_;
            uint64_t                        gen_  = 0;
            bool                            stop_ = false;
            std::atomic<int64_t>            dispatches_ {0};
        };

    } // namespace

    void dispatch(int nTasks, const std::function<void(int)> &task) {
        if (nTasks <= 1)
        {
            task(0);
            return;
        }
        Pool::instance().dispatch(nTasks, task);
    }

    bool inParallelRegion() noexcept {
        return tlInParallel;
    }

    int poolWorkers() noexcept {
        return Pool::instance().workerCount();
    }

    int64_t poolDispatches() noexcept {
        return Pool::instance().dispatchCount();
    }

}}} // namespace vknn::cpu::detail
