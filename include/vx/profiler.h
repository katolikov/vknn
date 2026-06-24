// vxrt — per-operator profiler (CPU wall + GPU timestamp), table/JSON/Chrome-trace export.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "vx/op.h"

namespace vx {

struct OpRecord {
  std::string name;
  OpType type = OpType::kUnknown;
  std::string backend;
  double cpuMs = 0.0;
  double gpuMs = -1.0;  // <0 => not measured
  std::array<uint32_t, 3> dispatch = {0, 0, 0};
  int64_t bytesIO = 0;
  bool fellBack = false;  // primary backend couldn't run -> CPU
};

class Profiler {
 public:
  void setEnabled(bool e) { enabled_ = e; }
  bool enabled() const { return enabled_; }
  void clear() { records_.clear(); }
  void add(const OpRecord& r) {
    if (enabled_) records_.push_back(r);
  }
  const std::vector<OpRecord>& records() const { return records_; }

  void printTable() const;  // sorted per-op timing table
  std::string toJson() const;
  void writeChromeTrace(const std::string& path) const;
  double totalCpuMs() const;
  double totalGpuMs() const;

 private:
  bool enabled_ = false;
  std::vector<OpRecord> records_;
};

}  // namespace vx
