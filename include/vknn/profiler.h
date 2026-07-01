// Per-op timing: CPU wall clock plus GPU timestamp queries. Prints a table, dumps JSON, and
// writes a chrome://tracing file.
#pragma once
#include "vknn/op_record.h"     // struct OpRecord
#include "vknn/profiler_class.h" // class Profiler (uses OpRecord)
