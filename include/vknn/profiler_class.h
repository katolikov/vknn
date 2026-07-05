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
        double totalGpuMs() const;

      private:
        bool                  enabled_ = false; ///< When false, add() records nothing.
        std::vector<OpRecord> records_;         ///< Collected records, in execution order.
    };

} // namespace vknn
