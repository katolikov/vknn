// Session holds the planned graph, the chosen backends and the caches. Runtime is the thin
// entry point callers actually use.
//
// Umbrella header: the public types live in one-per-file headers, included here in dependency
// order so every existing `#include "vknn/session.h"` keeps exposing the same names.
#pragma once
#include "vknn/io_tensor.h"     // struct IOTensor
#include "vknn/io_info.h"       // struct IOInfo
#include "vknn/session_class.h" // class Session (uses IOTensor, IOInfo)
#include "vknn/runtime.h"       // class Runtime (uses Session)
