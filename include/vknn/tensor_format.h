// Tensor-layout public surface: the TensorFormat enum plus the NCHW shape helpers.
//
// Umbrella header that re-exports the two pieces of the layout API so a caller includes one header
// instead of both. The IR is always NCHW; the Vulkan backend packs to the NC4HW4 boundary layout
// (channels grouped in vec4 blocks) internally.
//   - tensor_format_enum.h: the TensorFormat enum and formatStr().
//   - nchw.h:               the NCHW shape view, cBlocks()/formatElems(), and the kNC4Block constant.
#pragma once
#include "vknn/nchw.h"
#include "vknn/tensor_format_enum.h"
