#!/usr/bin/env bash
# Push VKNN binaries + model + assets to the device, run the classify example, pull results.
# Idempotent. Strict mode. Cleans up device temp on exit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEV_DIR="/data/local/tmp/vxrt"
BUILD_DIR="${BUILD_DIR:-build-android}"
BACKEND="${1:-vulkan}"        # vulkan | cpu
PRECISION="${2:-fp16}"        # fp16 | fp32

log() { printf '\033[36m>> %s\033[0m\n' "$*"; }

cleanup() { :; }   # leave caches/results on device for inspection; override if needed
trap cleanup EXIT

command -v adb >/dev/null || { echo "adb not found" >&2; exit 1; }
adb get-state >/dev/null 2>&1 || { echo "no device" >&2; exit 1; }

[[ -f "$BUILD_DIR/vknn_classify" ]] || { log "building"; ./scripts/build_android.sh; }
[[ -f assets/mobilenetv2.onnx ]] || { echo "missing assets/mobilenetv2.onnx (run scripts/get_golden.py)" >&2; exit 1; }
[[ -f assets/input.bin ]] || python3 scripts/get_golden.py --image assets/cat.jpg >/dev/null 2>&1 || true

log "pushing to $DEV_DIR"
adb shell "mkdir -p $DEV_DIR"
adb push "$BUILD_DIR/vknn_classify" "$BUILD_DIR/vknn_probe" "$BUILD_DIR/vknn_profile" \
         "$BUILD_DIR/vknn_backend_switch" "$BUILD_DIR/vknn_ion_zerocopy" "$DEV_DIR/" >/dev/null
adb push assets/mobilenetv2.onnx assets/input.bin assets/golden.bin "$DEV_DIR/" >/dev/null
adb shell "chmod +x $DEV_DIR/vknn_*"

log "running classify ($BACKEND, $PRECISION)"
adb shell "cd $DEV_DIR && ./vknn_classify --model mobilenetv2.onnx --input input.bin \
  --golden golden.bin --backend $BACKEND --precision $PRECISION --bench 30"

log "done. Binaries + caches remain in $DEV_DIR"
