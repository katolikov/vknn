// Conv kernel-selection + GPU-pass knobs: the Hint selector, the Mode value set, and the parser.
#pragma once
#include <string>

namespace vknn {

    /// Which knob a Mode value applies to (MNN-style Config::setHint). Autotune EFFORT is a separate
    /// top-level Config::tuning field (see tuning.h), not a Hint.
    enum class Hint {
        Winograd        = 0, ///< 3x3 Winograd selection (Auto / On / Off).
        WinogradVariant = 1, ///< Winograd matmul impl (TiledGemm / Fused / FusedSplit / FullyFused / SubgroupGemm).
        WinogradUnit    = 2, ///< Winograd output tile (F23 / F43 / F63).
        DirectConv3x3   = 3, ///< Direct 3x3 kernel (DirectAuto / RegisterTiled / LdsHalo).
        FlatLayout      = 4, ///< Flat row-major GPU layout pass — keeps generic head ops on the GPU (On / Off, default On).
        GpuIslandFold   = 5, ///< Fold tiny GPU op-islands onto the CPU (On / Off, default On).
        MatMulViewFold  = 6, ///< Fold Transpose/Expand chains into MatMul operand views at load (On / Off, default On).
        RopeFusion      = 7, ///< Fuse rotate-half RoPE chains into one Rope dispatch at load (On / Off, default On).
        FusedAttention  = 8, ///< Fuse the M=1 decode-attention chain into one FusedAttention kernel at load (On / Off, default On).
        KvConcatFold    = 9, ///< Fold the per-token KV-cache Concat into split-source FusedAttention reads at load (On / Off, default On). The rows-only present output it produces drives the engine-resident KV link like the cache-concat present (the link fold source comes from the present shape; see io_link.h).
        SplitKConv      = 10, ///< Split-K conv routing (Auto / On / Off, default Auto). Auto applies the
                              ///< calibrated deterministic shape rules (deep-reduction small-map convs run
                              ///< the split-K partial+reduce pair); On forces the general split-K on every
                              ///< structurally eligible KxK conv (fp16, batch 1, group 1, non-pointwise; an
                              ///< explicit DirectConv3x3 kernel force still wins); Off disables both the
                              ///< general and the 1x1 split-K paths.
        CoopmatGemm     = 11, ///< Cooperative-matrix MatMul routing (Auto / On / Off / Fp8 / Int8Coop,
                              ///< default Auto). Requires VK_KHR_cooperative_matrix with an enumerated
                              ///< subgroup-scope 16x16x16 row, pinnable 32-wide subgroups and the Vulkan
                              ///< memory model; a device without them runs the SSBO kernels unchanged at
                              ///< every value. Auto/On route eligible fp16 GEMMs (dense 2-D, batch 1,
                              ///< M,N multiples of 32, K multiple of 16, no view/bias/epilogue) through
                              ///< the fp16-operand fp32-accumulator coopmat kernel after a one-time
                              ///< on-device exact self-check. Fp8 / Int8Coop additionally quantize the
                              ///< A operand per-dispatch (per-tensor absmax scale) against a host-
                              ///< quantized weight operand - opt-in low-precision fast paths, never a
                              ///< default, numerics differ from the fp16 path by construction.
        KvCacheQuant    = 12, ///< Int8 KV-cache storage (Auto / On / Off, default Auto; Auto
                              ///< engages from kKvQuantAutoMinCacheBytes of eligible cache up,
                              ///< the measured size where the traffic saving beats the in-kernel
                              ///< dequantize). On stores the engine-resident decode KV
                              ///< cache as symmetric int8 with one fp16 scale per (token, head)
                              ///< row: the resident-link fold quantizes each present row on write
                              ///< and the FusedAttention kernels dequantize the past source inside
                              ///< their fp32 K-dot / V-apply loops (the current step's rows stay
                              ///< fp16). Halves cache memory and read traffic; numerics change,
                              ///< so Auto only engages above the measured size threshold. Requires the
                              ///< split-KV fold (Hint::KvConcatFold), fp16 storage, and 8-bit
                              ///< storage support — an ineligible model/device keeps the fp16
                              ///< cache byte-identically at every value (see src/core/kv_quant.h).
    };

    /// Every kernel/pass selection value, set uniformly via setHint(Hint, Mode). The value sets by
    /// Hint: Auto/On/Off serve Winograd, FlatLayout, GpuIslandFold, MatMulViewFold, RopeFusion,
    /// FusedAttention and KvConcatFold;
    /// TiledGemm..SubgroupGemm serve WinogradVariant; F23/F43/F63 serve WinogradUnit;
    /// DirectAuto..LdsHalo serve DirectConv3x3. The same underlying int recurs across groups
    /// (legal — the Hint picks the knob, the Mode the value). Forcing Winograd On/Off skips
    /// per-shape timing, making the choice deterministic run-to-run.
    enum class Mode {
        Auto          = 0,
        On            = 1,
        Off           = 2,
        TiledGemm     = 0,
        Fused         = 1,
        FusedSplit    = 2,
        FullyFused    = 3,
        SubgroupGemm  = 4,
        F23           = 0,
        F43           = 4,
        F63           = 6, ///< WinogradUnit: force F(6,3) (explicit-hint only; device measurement
                           ///< refuted promoting it into the automatic F(2,3)/F(4,3) rule).
        DirectAuto    = 0,
        RegisterTiled = 1,
        LdsHalo       = 2,
        Fp8           = 3, ///< CoopmatGemm: e4m3 operands, fp32 accumulation (opt-in).
        Int8Coop      = 4, ///< CoopmatGemm: int8 operands, int32 accumulation (opt-in).
    };

    /// Parse the Winograd knob from a string. Forcing on or off makes the 3x3-conv kernel choice
    /// deterministic run-to-run (no per-shape timing).
    /// @param s One of "auto" (measure per shape), "on" (force Winograd), "off" (force the direct kernel).
    /// @returns Mode::Auto, Mode::On, or Mode::Off respectively.
    Mode winogradFromStr(const std::string &s);

} // namespace vknn
