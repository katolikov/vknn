# CPU reference 3D Gaussian-Splatting rasterizer (gsplat 'classic' math) for YoNoSplat.
# This is the GOLDEN + the exact spec the from-scratch Vulkan compute rasterizer must match:
#   project mean -> 2D, project 3D world covariance -> 2D conic (+0.3 screen-space low-pass),
#   global depth sort, per-pixel front-to-back alpha compositing, SH degree-0 colour.
# Global depth sort + per-pixel composite is bit-identical (per pixel) to gsplat's per-tile sort:
# each pixel composites exactly the gaussians covering it, in depth order, either way.
#
# Inputs are the encoder's 6 Gaussian outputs (means/covariances/harmonics/opacities/rotations/
# scales) + the predicted extrinsics (camera-to-world) + the input intrinsics (normalized).
# Usage: ref_rasterizer.py <gaussians_dir> <extrinsics.bin> <intrinsics.bin> <out_dir> [--view N]
import sys, os, numpy as np

C0 = 0.28209479177387814  # SH band-0 constant
H = W = 224
NEAR = 0.2
TILE = 16

def load(p, shape, dt=np.float32):
    a = np.fromfile(p, dt).astype(np.float64)
    return a.reshape(shape)

def render_view(means, covs, colors, opac, w2c, K, H, W):
    """means [N,3] world, covs [N,3,3] world, colors [N,3], opac [N], w2c [4,4], K [3,3] pixel."""
    R, t = w2c[:3, :3], w2c[:3, 3]
    pc = means @ R.T + t                      # [N,3] camera space
    z = pc[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * pc[:, 0] / z + cx
    v = fy * pc[:, 1] / z + cy
    # 2D covariance: Sigma2d = J (R Sigma R^T) J^T + 0.3 I, J = perspective Jacobian (2x3)
    Scam = np.einsum('ij,njk,lk->nil', R, covs, R)          # [N,3,3]
    J = np.zeros((means.shape[0], 2, 3))
    J[:, 0, 0] = fx / z; J[:, 0, 2] = -fx * pc[:, 0] / (z * z)
    J[:, 1, 1] = fy / z; J[:, 1, 2] = -fy * pc[:, 1] / (z * z)
    cov2d = np.einsum('nij,njk,nlk->nil', J, Scam, J)        # [N,2,2]
    cov2d[:, 0, 0] += 0.3; cov2d[:, 1, 1] += 0.3
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    valid = (z > NEAR) & (det > 1e-12)
    det = np.where(det == 0, 1e-12, det)
    # conic = inverse(cov2d)
    conic = np.empty((means.shape[0], 3))                    # (a, b, c) for a*dx^2+2b*dx*dy+c*dy^2
    conic[:, 0] = cov2d[:, 1, 1] / det
    conic[:, 1] = -cov2d[:, 0, 1] / det
    conic[:, 2] = cov2d[:, 0, 0] / det
    # radius = 3 * sqrt(max eigenvalue)
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    lam = mid + np.sqrt(np.maximum(0.1, mid * mid - det))
    radius = np.ceil(3.0 * np.sqrt(lam)).astype(np.int64)

    img = np.zeros((H, W, 3))
    Tacc = np.ones((H, W))
    order = np.argsort(z)                                    # front-to-back (near first)
    for n in order:
        if not valid[n]:
            continue
        r = int(radius[n]); cu, cv = u[n], v[n]
        x0 = max(0, int(np.floor(cu - r))); x1 = min(W, int(np.ceil(cu + r)) + 1)
        y0 = max(0, int(np.floor(cv - r))); y1 = min(H, int(np.ceil(cv + r)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1); ys = np.arange(y0, y1)
        dx = xs[None, :] - cu; dy = ys[:, None] - cv
        a, b, c = conic[n]
        power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
        alpha = np.minimum(0.99, opac[n] * np.exp(np.minimum(power, 0.0)))
        da = np.where(alpha > (1.0 / 255.0), alpha, 0.0)   # gate out negligible alphas
        if not (da > 0).any():
            continue
        Tt = Tacc[y0:y1, x0:x1]
        w = da * Tt                                         # contribution = alpha * transmittance
        img[y0:y1, x0:x1] += w[..., None] * colors[n][None, None, :]
        Tacc[y0:y1, x0:x1] = Tt * (1.0 - da)                # transmittance *= (1 - alpha)
    return np.clip(img, 0.0, 1.0), 1.0 - Tacc


def main():
    gdir, extp, intp, outdir = sys.argv[1:5]
    view = int(sys.argv[sys.argv.index('--view') + 1]) if '--view' in sys.argv else 0
    os.makedirs(outdir, exist_ok=True)
    N = 100352
    means = load(f"{gdir}/means.bin" if os.path.exists(f"{gdir}/means.bin") else f"{gdir}/yono_means_gold.bin", (N, 3))
    covs = load(f"{gdir}/covariances.bin" if os.path.exists(f"{gdir}/covariances.bin") else f"{gdir}/yono_covariances_gold.bin", (N, 3, 3))
    harm = load(f"{gdir}/harmonics.bin" if os.path.exists(f"{gdir}/harmonics.bin") else f"{gdir}/yono_harmonics_gold.bin", (N, 3, 1))
    opac = load(f"{gdir}/opacities.bin" if os.path.exists(f"{gdir}/opacities.bin") else f"{gdir}/yono_opacities_gold.bin", (N,))
    colors = np.clip(C0 * harm[:, :, 0] + 0.5, 0.0, None)    # SH degree-0 -> RGB
    extr = load(extp, (2, 4, 4))                             # camera-to-world per view
    intr = load(intp, (2, 3, 3))
    Kpx = intr[view].copy(); Kpx[0] *= W; Kpx[1] *= H        # denormalize fx,cx and fy,cy
    w2c = np.linalg.inv(extr[view])
    img, alpha = render_view(means, covs, colors, opac, w2c, Kpx, H, W)
    (img * 255).astype(np.uint8).tofile(f"{outdir}/ref_view{view}_rgb.u8")
    np.save(f"{outdir}/ref_view{view}_rgb.npy", img.astype(np.float32))
    print(f"view {view}: rendered {img.shape} mean={img.mean():.4f} coverage(alpha>0.5)={float((alpha>0.5).mean()):.3f}")
    # also write a PNG if PIL is around
    try:
        from PIL import Image
        Image.fromarray((img * 255).astype(np.uint8)).save(f"{outdir}/ref_view{view}.png")
        print(f"  wrote {outdir}/ref_view{view}.png")
    except Exception as e:
        print("  (PIL not available, skipped PNG)", e)


if __name__ == "__main__":
    main()
