// Core of the flat-vector argmax epilogue (argmax_flat.comp / argmax_flat_fp16.comp): one
// workgroup strides the whole vector and reduces to the maximum element with first-occurrence
// (lowest index) tie semantics — the element a left-to-right host scan with a strictly-greater
// test selects. The Vulkan segment records this after the graph's dispatches on an output
// registered via Session::setOutputArgMax, so the host reads back {index, value} (8 bytes)
// instead of the full vector.
//
// SRC_ELEM_T = float / float16_t per variant. Values are widened to fp32 for the compare and the
// stored result; fp16 -> fp32 widening is exact and monotonic, so the selected index equals the
// index a host scan over the fp32-converted copy would pick, ties included. A NaN element never
// replaces the running maximum (compares false, as on the host); an all-NaN vector resolves to
// index 0.
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer SRC { SRC_ELEM_T sourceData[]; };
layout(std430, binding = 1) writeonly buffer RESULT { uint bestIndex; float bestValue; };
layout(push_constant) uniform PC { uint elemCount; } pc;

shared float laneValue[256];
shared uint  laneIndex[256];

void main() {
  const uint lane = gl_LocalInvocationID.x;
  float best   = -1.0 / 0.0; // -inf: any real element replaces it
  uint  bestAt = 0xFFFFFFFFu;
  for (uint i = lane; i < pc.elemCount; i += 256u) {
    const float v = float(sourceData[i]);
    if (v > best) { best = v; bestAt = i; }
  }
  laneValue[lane] = best;
  laneIndex[lane] = bestAt;
  barrier();
  // Tree reduce; lanes hold disjoint index sets, so "equal value -> lower index" is exactly the
  // global first-occurrence rule.
  for (uint activeLanes = 128u; activeLanes > 0u; activeLanes >>= 1u) {
    if (lane < activeLanes) {
      const float rightValue = laneValue[lane + activeLanes];
      const uint  rightIndex = laneIndex[lane + activeLanes];
      if (rightValue > laneValue[lane] || (rightValue == laneValue[lane] && rightIndex < laneIndex[lane])) {
        laneValue[lane] = rightValue;
        laneIndex[lane] = rightIndex;
      }
    }
    barrier();
  }
  if (lane == 0u) {
    bestIndex = laneIndex[0] == 0xFFFFFFFFu ? 0u : laneIndex[0];
    bestValue = laneValue[0];
  }
}
