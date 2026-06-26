"""Standalone harness: instantiate the YoNoSplat encoder (no hydra/lightning),
run a CPU forward, then export the encoder to ONNX.

Stage A (this run): just get a forward pass to succeed on dummy input.
"""
import sys, os, argparse, types as _types
sys.path.insert(0, "/tmp/YoNoSplat")

# The eager src/dataset/__init__.py pulls in the whole training/eval stack
# (lightning, dacite chains, ...) which the encoder model does NOT need.
# Stub the package with a real __path__ so only the lightweight submodules the
# encoder imports (dataset.types, dataset.shims.normalize_shim) load from disk.
for _pkg in ("src.dataset", "src.loss"):
    # src.loss/__init__ pulls in the CUDA splatting decoder (diff_gaussian_rasterization);
    # the encoder only needs src.loss.loss_pose.se3_inverse, which is self-contained.
    _m = _types.ModuleType(_pkg)
    _m.__path__ = ["/tmp/YoNoSplat/" + _pkg.replace(".", "/")]
    sys.modules[_pkg] = _m

import torch
import torch.nn as nn

# ---- build the config tree directly from the yaml-equivalent values ----
from src.model.encoder.encoder_yonosplat import (
    EncoderYoNoSplat, EncoderYoNoSplatCfg, OpacityMappingCfg,
)
from src.model.encoder.backbone.backbone_local_global import BackboneLocalGlobalCfg
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.encoder.visualization.encoder_visualizer_epipolar_cfg import (
    EncoderVisualizerEpipolarCfg,
)


def build_cfg():
    return EncoderYoNoSplatCfg(
        name="yonosplat",
        backbone=BackboneLocalGlobalCfg(
            name="local_global",
            intrinsics_embed_degree=4,
            intrinsics_embed_type="pixelwise",
            predict_intrinsics=True,
            use_pred_intrinsics_for_embed=False,
        ),
        visualizer=EncoderVisualizerEpipolarCfg(
            num_samples=8, min_resolution=256, export_ply=False
        ),
        gaussian_adapter=GaussianAdapterCfg(
            gaussian_scale_min=0.5, gaussian_scale_max=15.0, sh_degree=0
        ),
        opacity_mapping=OpacityMappingCfg(initial=0.0, final=0.0, warm_up=1),
        num_surfaces=1,
        input_mean=(0.0, 0.0, 0.0),
        input_std=(1.0, 1.0, 1.0),
        pretrained_weights="",   # don't load pi3; weights come from ckpt later
        pose_free=True,
        use_checkpoint=False,    # disable grad checkpointing for eval/export
        gaussian_downsample_ratio=1,
        gaussians_per_axis=14,
        upscale_token_ratio=2,
    )


_orig_inverse = torch.Tensor.inverse

def _inverse_patch(self, *a, **k):
    """Only a single batched 3x3 inverse (intrinsics, projection.py:84) fires on
    the encoder forward path. ONNX/vxrt have no Inverse op; use the exact analytic
    3x3 closed form (elementwise + stack). Fall back to original for other sizes."""
    if self.shape[-2:] == (3, 3):
        A = self
        a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
        d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
        g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        r0 = torch.stack([e * i - f * h, c * h - b * i, b * f - c * e], dim=-1)
        r1 = torch.stack([f * g - d * i, a * i - c * g, c * d - a * f], dim=-1)
        r2 = torch.stack([d * h - e * g, b * g - a * h, a * e - b * d], dim=-1)
        return torch.stack([r0, r1, r2], dim=-2) / det[..., None, None]
    return _orig_inverse(self, *a, **k)


_orig_diag_embed = torch.Tensor.diag_embed

def _diag_embed_patch(self, *a, **k):
    """torch.diag_embed (aten::diag_embed) -> exportable v.unsqueeze(-1)*eye(n)."""
    if not a and not k:  # plain vec(...,n) -> (...,n,n)
        n = self.shape[-1]
        I = torch.eye(n, device=self.device, dtype=self.dtype)
        return self.unsqueeze(-1) * I
    return _orig_diag_embed(self, *a, **k)


def patch_svd_orthogonalize():
    """CameraHead.svd_orthogonalize uses torch.svd (aten::svd, not ONNX/vxrt).
    It returns r = (orthogonal polar factor of normalize(m))^T. The polar factor
    is computed by Newton-Schulz iteration (matmuls only) -> exportable and
    GPU-friendly. For a trained pose head det(R)=+1, so this matches the SVD
    result (verified numerically against the golden full-forward outputs)."""
    import torch.nn.functional as F
    from src.model.encoder.layers.camera_head import CameraHead

    def _ns(self, m):
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        A = torch.transpose(F.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        X = A / (A.norm(dim=(-2, -1), keepdim=True) + 1e-12)
        for _ in range(18):  # Newton-Schulz -> orthogonal polar factor of A
            X = 1.5 * X - 0.5 * (X @ (X.transpose(-2, -1) @ X))
        return X.transpose(-2, -1)  # r = Q^T  (== v @ u^T in the SVD form)

    CameraHead.svd_orthogonalize = _ns


def patch_position_getter():
    """PositionGetter builds the integer patch-coordinate grid with
    torch.cartesian_prod (not ONNX-exportable). The grid is a pure constant for
    fixed h,w; build it with numpy so the graph sees a Constant, not cartesian_prod."""
    import numpy as np
    from src.model.encoder.layers.pos_embed import PositionGetter

    def _call(self, b, h, w, device=None):
        key = (int(h), int(w))
        if key not in self.cache_positions:
            ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            grid = torch.from_numpy(np.stack([ys.reshape(-1), xs.reshape(-1)], -1)).long()
            self.cache_positions[key] = grid
        pos = self.cache_positions[key]
        if device is not None:
            pos = pos.to(device)
        return pos.view(1, h * w, 2).expand(b, -1, 2).clone()

    PositionGetter.__call__ = _call


def bake_pos_embed(enc):
    """DINOv2 interpolates its pretrained pos-embed (37x37 -> 16x16) with
    antialiased bicubic every forward (aten::_upsample_bicubic2d_aa, not ONNX-
    exportable and not a vxrt GPU op). Input size is fixed (224), so the result
    is a constant: compute it once eagerly at full fidelity, then return the
    cached tensor so the traced graph sees a Constant."""
    vit = enc.backbone.encoder
    _orig = vit.interpolate_pos_encoding
    _cache = {}

    def _patched(self, x, w, h):
        key = (int(x.shape[1]), int(w), int(h))
        if key not in _cache:
            _cache[key] = _orig(x.detach(), w, h).detach()
        return _cache[key].to(x.dtype)

    vit.interpolate_pos_encoding = _types.MethodType(_patched, vit)


class EncoderWrapper(nn.Module):
    """tensors-in / tensors-out wrapper so torch.onnx.export sees a clean graph."""
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, image, intrinsics):
        g = self.enc({"image": image, "intrinsics": intrinsics}, global_step=0)
        return (g.means, g.covariances, g.harmonics, g.opacities, g.rotations, g.scales)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = build_cfg()
    print("[*] instantiating EncoderYoNoSplat (DINOv2 ViT-L/14-reg + 3 RoPE decoders)...")
    enc = EncoderYoNoSplat(cfg)
    enc.eval()
    n = sum(p.numel() for p in enc.parameters())
    print(f"[*] params: {n/1e6:.1f}M")

    if args.ckpt and os.path.exists(args.ckpt):
        print(f"[*] loading checkpoint {args.ckpt}")
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        # strip the lightning 'encoder.' prefix
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
        print(f"[*] loaded. missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:5]:
            print("    sample missing:", missing[:5])

    B, V, H, W = 1, args.views, 224, 224
    image = torch.rand(B, V, 3, H, W)
    K = torch.tensor([[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
    intrinsics = K.view(1, 1, 3, 3).expand(B, V, 3, 3).contiguous()

    torch.Tensor.inverse = _inverse_patch  # analytic 3x3 intrinsics inverse
    torch.Tensor.diag_embed = _diag_embed_patch  # exportable diag_embed
    patch_svd_orthogonalize()  # Newton-Schulz polar factor (no aten::svd)
    patch_position_getter()  # bake patch-coordinate grid (no cartesian_prod)
    bake_pos_embed(enc)  # remove the un-exportable bicubic-aa pos-embed Resize
    wrapper = EncoderWrapper(enc).eval()
    print(f"[*] running CPU forward, input image {tuple(image.shape)} ...")
    with torch.no_grad():
        outs = wrapper(image, intrinsics)
    names = ["means", "covariances", "harmonics", "opacities", "rotations", "scales"]
    print("[+] FORWARD OK. outputs:")
    for nm, t in zip(names, outs):
        print(f"      {nm:12} {tuple(t.shape)}  {t.dtype}")

    # save real reference I/O (golden) for ORT + later vxrt validation
    import numpy as np
    os.makedirs("/tmp/YoNoSplat/ref", exist_ok=True)
    np.save("/tmp/YoNoSplat/ref/image.npy", image.numpy())
    np.save("/tmp/YoNoSplat/ref/intrinsics.npy", intrinsics.numpy())
    for nm, t in zip(names, outs):
        np.save(f"/tmp/YoNoSplat/ref/{nm}.npy", t.numpy())
    print("[*] saved golden reference I/O to /tmp/YoNoSplat/ref/")

    if args.export:
        out_path = "/tmp/YoNoSplat/yonosplat_encoder.onnx"
        print(f"[*] exporting to ONNX -> {out_path} (opset 17, dynamo=False)...")
        with torch.no_grad():
            torch.onnx.export(
                wrapper, (image, intrinsics), out_path,
                input_names=["image", "intrinsics"], output_names=names,
                opset_version=17, dynamo=False, verbose=False,
            )
        print("[+] EXPORT OK")

        # numerical validation against onnxruntime
        print("[*] validating ONNX vs PyTorch in onnxruntime...")
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        ort_outs = sess.run(names, {
            "image": image.numpy(), "intrinsics": intrinsics.numpy()
        })
        print("[+] ORT ran. per-output cosine / max-abs-diff vs torch:")
        allok = True
        for nm, ot, tt in zip(names, ort_outs, outs):
            a = ot.reshape(-1).astype(np.float64)
            b = tt.numpy().reshape(-1).astype(np.float64)
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            mad = float(np.max(np.abs(a - b)))
            ok = cos > 0.9999
            allok = allok and ok
            print(f"      {nm:12} cos={cos:.6f}  max|Δ|={mad:.3e}  {'OK' if ok else 'FAIL'}")
        print("[+] ALL OUTPUTS MATCH" if allok else "[!] SOME OUTPUTS DIFFER")


if __name__ == "__main__":
    main()
