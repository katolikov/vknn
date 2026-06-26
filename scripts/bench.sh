#!/usr/bin/env bash
# Benchmark VKNN on-device across backends/precisions; report latency/fps + cold/warm session.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEV_DIR="/data/local/tmp/vxrt"
BUILD_DIR="${BUILD_DIR:-build-android}"
N="${1:-50}"
log() { printf '\033[36m>> %s\033[0m\n' "$*"; }

[[ -f "$BUILD_DIR/vknn_classify" ]] || ./scripts/build_android.sh
adb shell "mkdir -p $DEV_DIR"
adb push "$BUILD_DIR/vknn_classify" assets/mobilenetv2.onnx assets/input.bin assets/golden.bin "$DEV_DIR/" >/dev/null
adb shell "chmod +x $DEV_DIR/vknn_classify"

run() { # backend precision label
  adb shell "cd $DEV_DIR && ./vknn_classify --model mobilenetv2.onnx --input input.bin \
    --golden golden.bin --backend $1 --precision $2 --bench $N 2>/dev/null | grep -E 'golden compare|bench'" \
    | sed "s/^/  [$3] /"
}

log "cold vs warm session creation (Vulkan fp16)"
adb shell "cd $DEV_DIR && rm -rf cache && ./vknn_classify --model mobilenetv2.onnx --input input.bin \
  --backend vulkan --precision fp16 2>&1 | grep 'Session created'" | sed 's/^/  COLD /'
adb shell "cd $DEV_DIR && ./vknn_classify --model mobilenetv2.onnx --input input.bin \
  --backend vulkan --precision fp16 2>&1 | grep 'Session created'" | sed 's/^/  WARM /'

log "inference latency ($N timed runs each)"
run vulkan fp16 "Vulkan fp16"
run vulkan fp32 "Vulkan fp32"
run cpu    fp32 "CPU-NEON"
log "done"
