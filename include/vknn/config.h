// Runtime config: a plain struct of knobs plus a JSON loader. Field set mirrors MNN's config.
// Every field is documented in docs/CONFIG.md.
//
// Umbrella header: the config types now live one-per-file. Include this to pull them all in.
#pragma once
#include "vknn/common.h"
#include "vknn/tensor_format.h"

#include "vknn/backend_kind.h"    // BackendKind + backendName/backendFromStr
#include "vknn/precision.h"       // Precision + mixedPrecisionFp32Tensors/precisionFromStr
#include "vknn/tuning.h"          // Tuning + tuningFromStr
#include "vknn/hint.h"            // Hint + Mode + winogradFromStr
#include "vknn/config_struct.h"   // Config (depends on all of the above)
