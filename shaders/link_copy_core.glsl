// Core of the resident-link copy kernel (link_copy.comp / link_copy_fp16.comp): moves caller-declared
// canonical element ranges from a linked boundary OUTPUT buffer into a linked boundary INPUT buffer at
// the head of the segment's command stream, before any node reads the input — the on-device fold of a
// recurrent state (e.g. present KV rows into the past cache), replacing the per-run host round trip.
//
// Elements move as integer words (LINK_ELEM_T = uint / uint16_t per variant): a bit copy, no float
// semantics, so the destination receives exactly the source's stored bits. Range offsets are CANONICAL
// (logical NCHW row-major) element positions; deviceIndex() maps them to the buffer layout — identity
// for a flat tensor, the channel-block interleave for NC4HW4 (matching packToBuffer's index math). The
// ranges SSBO is host-updated between runs, so the dispatch is recorded once: a fixed grid strides over
// totalElems, and a zero header (no ranges yet) makes the dispatch a no-op.
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer SRC { LINK_ELEM_T sourceData[]; };
layout(std430, binding = 1) buffer DST { LINK_ELEM_T destData[]; };
// spans holds 3 uints per range: {sourceOffset, destOffset, count} in canonical elements.
layout(std430, binding = 2) readonly buffer RANGES { uint rangeCount; uint totalElems; uint spans[]; };
layout(push_constant) uniform PC {
  int srcC, srcH, srcW;
  int dstC, dstH, dstW;
  int srcFmt, dstFmt; // 0 = flat row-major, 2 = NC4HW4 (codes match boundary_convert.comp)
} pc;

// Buffer element index for canonical (NCHW row-major) position `canon` under layout `fmt`.
int deviceIndex(int fmt, int canon, int C, int H, int W) {
  if (fmt == 2) { // NC4HW4: channels in blocks of four, lane = c % 4
    int w = canon % W; int t = canon / W;
    int h = t % H; t /= H;
    int c = t % C; int n = t / C;
    int Cb = (C + 3) / 4;
    return ((((n * Cb + c / 4) * H + h) * W + w) * 4) + (c % 4);
  }
  return canon;
}

void main() {
  uint gid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;
  uint stride = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
  for (uint i = gid; i < totalElems; i += stride) {
    uint base = 0;
    for (uint r = 0; r < rangeCount; ++r) {
      uint len = spans[r * 3u + 2u];
      if (i - base < len) {
        int srcCanon = int(spans[r * 3u + 0u] + (i - base));
        int dstCanon = int(spans[r * 3u + 1u] + (i - base));
        destData[deviceIndex(pc.dstFmt, dstCanon, pc.dstC, pc.dstH, pc.dstW)] =
            sourceData[deviceIndex(pc.srcFmt, srcCanon, pc.srcC, pc.srcH, pc.srcW)];
        break;
      }
      base += len;
    }
  }
}
