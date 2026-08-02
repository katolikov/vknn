// Reduce wire codes and the vec4 combine the blocked reduction family shares.
//
// The codes mirror vknn::ReduceType (include/vknn/reduce_type.h) and match the flat family's
// spelling in flat_reduce.comp, so one node's sub-op selects the same arithmetic on either path.
// The blocked kernels reduce four channels at a time, hence the vec4 forms: a channel block's four
// lanes are independent reductions that happen to travel together.
#ifndef VKNN_NC4_REDUCE_CODES
#define VKNN_NC4_REDUCE_CODES

#define REDUCE_MEAN 0
#define REDUCE_SUM  1
#define REDUCE_MAX  2
#define REDUCE_MIN  3
#define REDUCE_PROD 4
#define REDUCE_L2   5

// The value a reduction starts from, so an empty slice folds to the neutral element.
vec4 vxReduceIdentity(int op) {
  if (op == REDUCE_MAX)  return vec4(VX_NEG_INF);
  if (op == REDUCE_MIN)  return vec4(VX_POS_INF);
  if (op == REDUCE_PROD) return vec4(1.0);
  return vec4(0.0); // SUM / MEAN / L2 accumulate from zero
}

// Fold one loaded quad into the accumulator. L2 squares on the way in and takes its root at the end.
vec4 vxReduceStep(int op, vec4 acc, vec4 v) {
  if (op == REDUCE_MAX)  return max(acc, v);
  if (op == REDUCE_MIN)  return min(acc, v);
  if (op == REDUCE_PROD) return acc * v;
  if (op == REDUCE_L2)   return acc + v * v;
  return acc + v;
}

// Combine two partial accumulators. Same operation as a step for every associative code; the split
// exists because a partial already carries squared values for L2, which must not be squared twice.
vec4 vxReduceMerge(int op, vec4 a, vec4 b) {
  if (op == REDUCE_MAX)  return max(a, b);
  if (op == REDUCE_MIN)  return min(a, b);
  if (op == REDUCE_PROD) return a * b;
  return a + b;
}

// Turn the folded accumulator into the reduction's value over `count` elements.
vec4 vxReduceFinish(int op, vec4 acc, int count) {
  if (op == REDUCE_MEAN) return count > 0 ? acc / float(count) : acc;
  if (op == REDUCE_L2)   return sqrt(acc);
  return acc;
}

#endif
