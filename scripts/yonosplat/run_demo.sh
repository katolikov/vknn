#!/usr/bin/env bash
# Run the full YoNoSplat 3DGS pipeline end-to-end on the device GPU, in ONE program:
#   image + intrinsics -> encoder (VKNN Vulkan) -> Gaussians -> Vulkan rasterizer -> rendered view (PPM).
# Prereqs on device (/data/local/tmp/vxrt/yono): encoder_fp16.vxm, yono_image.bin, yono_intrinsics.bin.
# Build + push first:  ./scripts/build_android.sh && adb push build-android/vknn_yonosplat /data/local/tmp/vxrt/
set -e
DIR=/data/local/tmp/vxrt
adb push build-android/vknn_yonosplat $DIR/vknn_yonosplat >/dev/null
adb shell chmod +x $DIR/vknn_yonosplat

# Optional: dump the encoder's predicted camera pose so the rasterizer renders the input view
# (otherwise it renders from an identity camera). One extra encoder run.
adb shell "cd $DIR/yono && rm -rf dump && mkdir -p dump && \
  VXRT_DUMP_NAMES=/enc/Reshape_7_output_0 ../vknn_run_io encoder_fp16.vxm out --no-weight-cache \
  yono_image.bin yono_intrinsics.bin >/dev/null 2>&1 || true"

echo '== full pipeline: image -> encoder(GPU) -> Gaussians -> rasterizer(GPU) -> view =='
adb shell "cd $DIR/yono && ../vknn_yonosplat encoder_fp16.vxm yono_image.bin yono_intrinsics.bin \
  render.ppm --extr dump/_enc_Reshape_7_output_0.bin --view 0"
adb pull $DIR/yono/render.ppm /tmp/yono_render.ppm
echo "pulled -> /tmp/yono_render.ppm"
