// Shared GridSample tap machinery: coordinate padding (handle/reflectc), the cubic weight fan,
// and the per-pixel tap resolver. Every gridsample*.comp variant includes this file after its PC
// block (the helpers read pc.Win/pc.Hin/pc.align and the MODE/PADMODE specialization constants),
// so the sampler geometry has one definition across the plain/warp and fp32/fp16 variants — a
// change applied to only one of them is the silent-wrong-answer class the shared lane-count
// contract in gridsample.cpp guards against.

// An out-of-range tap axis: the resolvers return it instead of a column index / row offset, and
// the variant's tapAt() reads any tap with an OOB axis as vec4(0). Mirrors kGridSampleTapOob in
// src/backend/vulkan/ops/gridsample_rule.h, which holds the host definition of everything below.
#define GS_TAP_OOB (-1)

// A source plane with a zero extent holds no pixel to sample. Mirrors gridSampleSourceEmpty().
bool gridSampleSourceEmpty() { return pc.Hin <= 0 || pc.Win <= 0; }
// Whether a resolved tap can carry GS_TAP_OOB, i.e. whether the variant's tapAt() has to test for
// it: the zeros mode reports out-of-range taps by construction, and an empty source plane has no
// in-range tap in ANY padding mode. Both terms are invariant across the dispatch (PADMODE is a
// specialization constant, the extents are push constants), so the test folds away for a
// border/reflection variant sampling a non-empty plane and the per-tap cost stays one add + load.
// Mirrors gridSampleTapsCanBeOob().
bool gridSampleTapsCanBeOob() { return PADMODE == 0 || gridSampleSourceEmpty(); }

float reflectc(float x, float lo, float hi) {
  if (hi <= lo) return lo;
  float rng = hi - lo, t = mod(x - lo, 2.0 * rng);
  if (t < 0.0) t += 2.0 * rng;
  if (t > rng) t = 2.0 * rng - t;
  return lo + t;
}
float handle(float c, int S) {
  if (PADMODE == 2) return reflectc(c, pc.align == 1 ? 0.0 : -0.5, pc.align == 1 ? float(S - 1) : float(S) - 0.5);
  return c;
}
// Per-axis tap resolution, applied ONCE PER PIXEL: the tap grid is separable (px and py resolve
// independently in every padding mode), so the per-pixel state is one small array per axis and
// the channel-block loop pays a single add per tap — no padding or address math per block.
// Reflection reflects each tap (cubic taps reach +-2 px past the mapped coordinate, where a
// clamp diverges from the reflected pixel; for linear/nearest the two agree).
// resolveTapX yields a column index, resolveTapRow a row offset (py * Win); an axis with no
// in-range pixel yields GS_TAP_OOB — the zeros mode's out-of-range taps, and an empty source
// plane in every mode.
int resolveTapX(int px) {
  // No column exists, in any padding mode: the border/reflection clamp below would run over the
  // empty range [0, -1] — undefined — and the tap would index past an empty source.
  if (pc.Win <= 0) return GS_TAP_OOB;
  if (PADMODE == 0) {
    if (px < 0 || px >= pc.Win) return GS_TAP_OOB;
  } else if (PADMODE == 2) {
    px = clamp(int(round(reflectc(float(px), pc.align == 1 ? 0.0 : -0.5, pc.align == 1 ? float(pc.Win - 1) : float(pc.Win) - 0.5))), 0, pc.Win - 1);
  } else {
    px = clamp(px, 0, pc.Win - 1);
  }
  return px;
}
int resolveTapRow(int py) {
  if (pc.Hin <= 0) return GS_TAP_OOB; // no row exists, in any padding mode (see resolveTapX)
  if (PADMODE == 0) {
    if (py < 0 || py >= pc.Hin) return GS_TAP_OOB;
  } else if (PADMODE == 2) {
    py = clamp(int(round(reflectc(float(py), pc.align == 1 ? 0.0 : -0.5, pc.align == 1 ? float(pc.Hin - 1) : float(pc.Hin) - 0.5))), 0, pc.Hin - 1);
  } else {
    py = clamp(py, 0, pc.Hin - 1);
  }
  return py * pc.Win;
}
// Cubic tap-fan geometry: 4 taps per axis at floor-1..floor+2.
#define GS_CUBIC_TAPS 4

// Cubic convolution weights (alpha = -0.75): taps at floor-1..floor+2 per axis.
void cubicW(float t, out float w[GS_CUBIC_TAPS]) {
  const float A = -0.75;
  float t1 = 1.0 + t, t2 = 2.0 - t;
  w[0] = A * t1 * t1 * t1 - 5.0 * A * t1 * t1 + 8.0 * A * t1 - 4.0 * A;
  w[1] = (A + 2.0) * t * t * t - (A + 3.0) * t * t + 1.0;
  w[2] = (A + 2.0) * (1.0 - t) * (1.0 - t) * (1.0 - t) - (A + 3.0) * (1.0 - t) * (1.0 - t) + 1.0;
  w[3] = A * t2 * t2 * t2 - 5.0 * A * t2 * t2 + 8.0 * A * t2 - 4.0 * A;
}
