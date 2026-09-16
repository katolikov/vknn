// Boolean reading shared by the flat logical kernels (or.comp, xor.comp, not.comp).
//
// An operand is true iff it is nonzero -- the CPU oracle's rule (src/backend/cpu/logical_ops.h): NaN,
// infinities and subnormals are true, +0.0 and -0.0 are false. The test runs on the IEEE-754 bit
// pattern instead of a float compare, so its answer never depends on how a driver evaluates an
// unordered (NaN) comparison: dropping the sign bit leaves an all-zero word exactly for the two zeros.
// An fp16-stored operand widens to fp32 first; the widening maps the zeros to zeros and every other
// half (NaN and subnormals included) to a nonzero float, so the rule is the same at both precisions.
#ifndef VKNN_LOGICAL_TRUTH_GLSL
#define VKNN_LOGICAL_TRUTH_GLSL

// Left shift that discards bit 31, the sign bit of a 32-bit IEEE-754 pattern.
const uint kLogicalSignBitDrop = 1u;
// Canonical boolean results (the compare family's 1.0 / 0.0), exact at fp16 and fp32 storage.
const float kLogicalTrue  = 1.0;
const float kLogicalFalse = 0.0;

bool logicalIsTrue(float value) {
  return (floatBitsToUint(value) << kLogicalSignBitDrop) != 0u;
}

#endif  // VKNN_LOGICAL_TRUTH_GLSL
