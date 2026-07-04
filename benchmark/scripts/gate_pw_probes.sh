#!/bin/bash
# Device gate for producer-epilogue fusion: per probe, compile fused+nofuse (fp32+fp16), run on
# device, byte-compare outputs, and fail on any CPU fallback in the fused runs.
#
#   benchmark/scripts/make_pw_probes.py /tmp/pwprobe     # generate the probe ONNX + input set
#   benchmark/scripts/gate_pw_probes.sh [compile|push|run|all] [probes-dir] [device-dir]
set -u
cd "$(dirname "$0")/../.."
PROBES_DIR=${2:-/tmp/pwprobe}
DEV=${3:-/data/local/tmp/pw/probes}
COMPILE=./build-host/vknn_compile

PROBES="p_conv p_conv1x1 p_dwconv p_convtr p_softmax p_layernorm p_reduce p_gridsample p_resize p_avgpool p_maxpool p_gap p_gemm p_matmul p_conv1x1s2 p_conv1x1deep p_conv3x3w p_softmax_nc4 p_gemm_nobias p_layernorm_nobeta p_reduce_attr p_maxpool_full"

phase="${1:-all}"

if [ "$phase" = "compile" ] || [ "$phase" = "all" ]; then
  echo "== compile =="
  for p in $PROBES; do
    for cfg in "f32:" "n32:--no-fuse-pointwise" "f16:--fp16" "n16:--fp16 --no-fuse-pointwise"; do
      tag="${cfg%%:*}"; flags="${cfg#*:}"
      out=$($COMPILE $PROBES_DIR/$p.onnx $PROBES_DIR/${p}_$tag.vxm $flags 2>&1)
      fuse=$(echo "$out" | grep -o "fused [0-9]* chain(s), [0-9]* into producer epilogues" | head -1)
      if [ "$tag" = "f32" ]; then echo "$p: ${fuse:-no chains}"; fi
      if echo "$out" | grep -qi "error"; then echo "COMPILE FAIL $p $tag"; echo "$out" | tail -3; fi
    done
  done
fi

if [ "$phase" = "push" ] || [ "$phase" = "all" ]; then
  echo "== push =="
  adb shell "mkdir -p $DEV" > /dev/null
  adb push $PROBES_DIR/*.vxm $PROBES_DIR/*_in.bin $PROBES_DIR/p_gridsample_grid.bin $DEV/ 2>&1 | tail -1
  adb push build-android/vknn_run_io /data/local/tmp/pw/vknn_run_io 2>&1 | tail -1
fi

if [ "$phase" = "run" ] || [ "$phase" = "all" ]; then
  echo "== run + compare =="
  PASS=0; FAIL=0
  for p in $PROBES; do
    ins="${p}_in.bin"
    if [ "$p" = "p_gridsample" ]; then ins="${p}_in.bin p_gridsample_grid.bin"; fi
    for prec in 32 16; do
      pflag="high"; [ "$prec" = "16" ] && pflag="low"
      wflag="off"
      # extra wino run handled below
      r=$(adb shell "cd $DEV && rm -rf o_${p}_f$prec o_${p}_n$prec; ../vknn_run_io ${p}_f$prec.vxm o_${p}_f$prec --backend vulkan --precision $pflag --no-cache --no-fold-islands --winograd $wflag --tuning none --cache . $ins > l_${p}_f$prec.log 2>&1; ../vknn_run_io ${p}_n$prec.vxm o_${p}_n$prec --backend vulkan --precision $pflag --no-cache --no-fold-islands --winograd $wflag --tuning none --cache . $ins > l_${p}_n$prec.log 2>&1; nf=\$(ls o_${p}_f$prec 2>/dev/null | wc -l); ok=1; [ \$nf -lt 1 ] && ok=0; for f in \$(ls o_${p}_f$prec 2>/dev/null); do cmp o_${p}_f$prec/\$f o_${p}_n$prec/\$f > /dev/null 2>&1 || ok=0; done; [ \$ok = 1 ] && echo CMP_OK || echo CMP_DIFF; grep -ci 'falling back' l_${p}_f$prec.log")
      ok=$(echo "$r" | grep -c CMP_OK); fb=$(echo "$r" | tail -1)
      if [ "$ok" = "1" ] && [ "$fb" = "0" ]; then
        echo "PASS $p fp$prec"; PASS=$((PASS+1))
      else
        echo "FAIL $p fp$prec  ($r)" ; FAIL=$((FAIL+1))
      fi
    done
  done
  # Winograd path: p_conv3x3w fp16 with --winograd on
  r=$(adb shell "cd $DEV && rm -rf o_w_f o_w_n; ../vknn_run_io p_conv3x3w_f16.vxm o_w_f --backend vulkan --precision low --no-cache --no-fold-islands --winograd on --tuning none --cache . p_conv3x3w_in.bin > l_w_f.log 2>&1; ../vknn_run_io p_conv3x3w_n16.vxm o_w_n --backend vulkan --precision low --no-cache --no-fold-islands --winograd on --tuning none --cache . p_conv3x3w_in.bin > l_w_n.log 2>&1; nf=\$(ls o_w_f 2>/dev/null | wc -l); ok=1; [ \$nf -lt 1 ] && ok=0; for f in \$(ls o_w_f 2>/dev/null); do cmp o_w_f/\$f o_w_n/\$f > /dev/null 2>&1 || ok=0; done; [ \$ok = 1 ] && echo CMP_OK || echo CMP_DIFF; grep -ci 'falling back' l_w_f.log")
  ok=$(echo "$r" | grep -c CMP_OK); fb=$(echo "$r" | tail -1)
  if [ "$ok" = "1" ] && [ "$fb" = "0" ]; then echo "PASS p_conv3x3w fp16 winograd-on"; PASS=$((PASS+1)); else echo "FAIL p_conv3x3w fp16 winograd-on ($r)"; FAIL=$((FAIL+1)); fi
  echo "== $PASS passed, $FAIL failed =="
fi
