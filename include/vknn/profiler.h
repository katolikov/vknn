/// @file
/// Public umbrella header for the per-op profiler. Including this pulls in both the profiler and
/// its record type as one unit; nothing is declared here directly.
///
/// The profiler collects one OpRecord per executed op — CPU wall clock plus GPU timestamp-query
/// time — and renders that timeline as a sorted table, a JSON dump, or a chrome://tracing file.
#pragma once
#include "vknn/op_record.h"      // struct OpRecord
#include "vknn/profiler_class.h" // class Profiler (uses OpRecord)
