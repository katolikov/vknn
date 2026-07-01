// Two views of a tensor: TensorDesc is the static description in the graph, RtTensor is the
// live thing during a run (host data, device data, or both, with validity flags).
#pragma once
#include "vknn/common.h"
#include "vknn/dtype.h"
#include "vknn/op.h"
#include "vknn/tensor_format.h"
#include <cstring>
#include <memory>
#include <vector>

// Per-type headers, in dependency order (a type used by another comes first).
#include "vknn/tensor_desc.h"     // struct TensorDesc
#include "vknn/host_buffer.h"     // struct HostBuffer
#include "vknn/device_storage.h"  // struct DeviceStorage (forward declaration)
#include "vknn/rt_tensor.h"       // struct RtTensor (uses HostBuffer + DeviceStorage)
