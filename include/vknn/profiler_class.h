// Per-op profiler: collects OpRecords, prints a table, dumps JSON, and writes a chrome://tracing file.
#pragma once
#include "vknn/op_record.h"
#include <string>
#include <vector>

namespace vknn {

    class Profiler {
      public:
        void setEnabled(bool e) {
            enabled_ = e;
        }
        bool enabled() const {
            return enabled_;
        }
        void clear() {
            records_.clear();
        }
        void add(const OpRecord &r) {
            if (enabled_)
            {
                records_.push_back(r);
            }
        }
        const std::vector<OpRecord> &records() const {
            return records_;
        }

        void        printTable() const; // sorted per-op timing table
        std::string toJson() const;
        void        writeChromeTrace(const std::string &path) const;
        double      totalCpuMs() const;
        double      totalGpuMs() const;

      private:
        bool                  enabled_ = false;
        std::vector<OpRecord> records_;
    };

} // namespace vknn
