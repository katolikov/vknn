// Compiler-portability macros shared across the engine, backends, and tests.
#pragma once

// Force a function out-of-line. The CPU float oracles pin their accumulation order by keeping the
// loop in ONE compiled body: inlining would let the optimizer specialize the loop per call site and
// fuse a*b+acc in some clones but not others, so equal inputs could round differently between
// calls. MSVC spells the attribute __declspec(noinline); clang-cl accepts either form.
#if defined(_MSC_VER) && !defined(__clang__)
#define VKNN_NOINLINE __declspec(noinline)
#else
#define VKNN_NOINLINE __attribute__((noinline))
#endif
