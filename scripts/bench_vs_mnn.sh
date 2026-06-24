#!/usr/bin/env bash
# Head-to-head: vxrt-Vulkan vs MNN-Vulkan on the connected device, both at their fastest
# (Vulkan backend, fp16). Honest, reproducible. Expects MNN already built+converted (see the
# "SETUP" notes below); if the MNN bits are missing it just runs vxrt and says so.
#
# SETUP (one time, on the host):
#   git clone --depth 1 https://github.com/alibaba/MNN /tmp/MNN
#   # Android lib + runner (Vulkan on):
#   cmake -S /tmp/MNN -B /tmp/MNN/build-android -GNinja \
#     -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33 -DMNN_VULKAN=ON \
#     -DMNN_BUILD_BENCHMARK=ON -DMNN_SEP_BUILD=OFF -DCMAKE_BUILD_TYPE=Release
#   ninja -C /tmp/MNN/build-android MNN MNNV2Basic.out
#   # Host converter, then convert the model to fp16 .mnn:
#   cmake -S /tmp/MNN -B /tmp/MNN/build-host -GNinja -DMNN_BUILD_CONVERTER=ON -DCMAKE_BUILD_TYPE=Release
#   ninja -C /tmp/MNN/build-host MNNConvert
#   /tmp/MNN/build-host/MNNConvert -f ONNX --modelFile assets/mobilenetv2.onnx \
#     --MNNModel /tmp/mobilenetv2_fp16.mnn --fp16
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
DEV=/data/local/tmp/vxrt
MNN_OUT=/tmp/MNN/build-android/MNNV2Basic.out
MNN_LIB=/tmp/MNN/build-android/OFF/arm64-v8a/libMNN.so
MNN_MODEL=/tmp/mobilenetv2_fp16.mnn
LOOPS="${1:-30}"
log() { printf '\033[36m>> %s\033[0m\n' "$*"; }

[[ -f build-android/vx_classify ]] || ./scripts/build_android.sh
adb shell "mkdir -p $DEV $DEV/mnn"
adb push build-android/vx_classify assets/mobilenetv2.onnx assets/input.bin assets/golden.bin "$DEV/" >/dev/null
adb shell "chmod +x $DEV/vx_classify"

log "vxrt — Vulkan fp16 (warm, autotuned)"
adb shell "cd $DEV && ./vx_classify --model mobilenetv2.onnx --input input.bin --golden golden.bin \
  --backend vulkan --precision fp16 --bench $LOOPS 2>/dev/null | grep -iE 'golden compare|bench'"

if [[ -f "$MNN_OUT" && -f "$MNN_LIB" && -f "$MNN_MODEL" ]]; then
  adb push "$MNN_OUT" "$MNN_LIB" "$MNN_MODEL" "$DEV/mnn/" >/dev/null
  adb shell "chmod +x $DEV/mnn/MNNV2Basic.out"
  log "MNN — Vulkan fp16 (forwardType=7, precision=Low); warming cache first"
  adb shell "cd $DEV/mnn && LD_LIBRARY_PATH=. ./MNNV2Basic.out $(basename "$MNN_MODEL") $LOOPS 0 7 1 2 1x3x224x224 2>/dev/null | grep -i Avg" || true
  adb shell "cd $DEV/mnn && LD_LIBRARY_PATH=. ./MNNV2Basic.out $(basename "$MNN_MODEL") $LOOPS 0 7 1 2 1x3x224x224 2>/dev/null | grep -i Avg"
  log "MNN — CPU 4-thread fp16 (reference; MNN's optimized CPU backend)"
  adb shell "cd $DEV/mnn && LD_LIBRARY_PATH=. ./MNNV2Basic.out $(basename "$MNN_MODEL") $LOOPS 0 0 4 2 1x3x224x224 2>/dev/null | grep -i Avg"
else
  log "MNN artifacts not found (see SETUP at top); skipping MNN side."
fi
