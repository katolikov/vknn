// Backend interface, the segment execution model, and the backend registry.
//
// The Session walks the topo-sorted nodes and groups consecutive nodes that landed on the same
// backend into a "segment". Each backend turns its segment into a Segment object: the Vulkan
// backend records one command buffer for the whole thing, the CPU backend keeps a list of ops.
// Segments run in order; a tensor crossing a backend boundary is synced there (toHost/toDevice).
// This yields a single pre-recorded GPU submit for the common case plus a CPU fallback for ops the
// GPU cannot run, with minimal copying.
//
// Umbrella header: each top-level type now lives in its own header (one type per file). This file
// re-exports them, in dependency order, so `#include "vknn/backend.h"` keeps exposing the same names.
#pragma once
#include "vknn/exec_context.h"     // struct ExecContext (+ fwd class Profiler)
#include "vknn/backend_class.h"    // class Backend (fwd-decls class Segment)
#include "vknn/segment.h"          // class Segment (fwd-decls class Backend)
#include "vknn/backend_registry.h" // class BackendRegistry, struct BackendRegistrar, VKNN_REGISTER_BACKEND
