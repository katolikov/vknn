// Tensor layouts. The IR is always NCHW; the Vulkan backend packs to NC4HW4 internally.
#pragma once
#include "vknn/tensor_format_enum.h"
#include "vknn/nchw.h"
