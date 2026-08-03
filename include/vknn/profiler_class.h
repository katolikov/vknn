// Per-op profiler: collects OpRecords, prints a table, dumps JSON, and writes a chrome://tracing file.
#pragma once
#include "vknn/op_record.h"
#include <string>
#include <vector>

namespace vknn {

    /// Collects one OpRecord per executed op and reports the resulting timings. Recording is off by
    /// default; when disabled add() is a no-op, so an enclosing engine can leave calls in place with no
    /// overhead. The accumulated records feed a sorted console table, a JSON dump, a chrome://tracing
    /// file, and CPU/GPU wall-time totals.
    class Profiler {
      public:
        /// Turn recording on or off. While off, add() drops its argument and records() stays as it was.
        void setEnabled(bool e) noexcept {
            enabled_ = e;
        }
        /// True when add() is currently recording.
        bool enabled() const noexcept {
            return enabled_;
        }
        /// Drop all accumulated records, leaving the enabled state unchanged.
        void clear() noexcept {
            records_.clear();
            gpuSpanMs_ = 0;
        }
        /// Append a copy of `r` when recording is enabled; a no-op otherwise.
        void add(const OpRecord &r) {
            if (enabled_)
            {
                records_.push_back(r);
            }
        }
        /// The records collected so far, in execution order.
        const std::vector<OpRecord> &records() const noexcept {
            return records_;
        }

        /// Print a per-node timing table to stdout followed by a per-op-type summary sorted by GPU time.
        void printTable() const;
        /// Serialize the records as a JSON array (one object per record).
        std::string toJson() const;
        /// Write a chrome://tracing JSON file at `path`: one complete event per record on a single
        /// synthetic timeline. Silently does nothing if the file cannot be opened.
        void writeChromeTrace(const std::string &path) const;
        /// Sum of every record's CPU wall-clock time, in milliseconds.
        double totalCpuMs() const;
        /// Sum of the measured GPU timestamp times, in milliseconds; records with no GPU measurement
        /// (negative gpuMs) are skipped.
        ///
        /// This is NOT the wall-clock GPU time of the run: a node's record brackets its commands
        /// with two timestamps, and the GPU overlaps consecutive nodes, so the intervals overlap and
        /// their sum exceeds the elapsed span. A node that issues no commands of its own (an op the
        /// planner elided to a zero-copy view) still measures whatever the GPU was draining between
        /// its two timestamps, which is how an elided Concat came to carry 28% of a classifier's
        /// reported time while dispatching nothing. Use it to RANK ops, and gpuSpanMs() for elapsed
        /// time.
        double totalGpuMs() const;
        /// Elapsed GPU time of the run: first command started to last command finished, as the
        /// backend measured it. This is the figure the published benchmarks quote. Zero when the
        /// backend recorded no span (no GPU segment ran, or profiling produced no timestamps).
        double gpuSpanMs() const noexcept {
            return gpuSpanMs_;
        }
        /// Record the elapsed GPU span for the run; the backend calls this once per segment, and the
        /// spans of several segments add up to the run's.
        void addGpuSpanMs(double ms) noexcept {
            if (enabled_ && ms > 0)
            {
                gpuSpanMs_ += ms;
            }
        }
        /// Sum of every record's recorded dispatch count. Covers the ops only — a segment's
        /// boundary/epilogue dispatches belong to no record, so this is at or below the segment's
        /// own dispatch total (which the segment logs).
        uint64_t totalDispatches() const;

      private:
        bool                  enabled_ = false; ///< When false, add() records nothing.
        std::vector<OpRecord> records_;         ///< Collected records, in execution order.
        double                gpuSpanMs_ = 0;   ///< Elapsed GPU time, summed over the segments that ran.
    };

} // namespace vknn
