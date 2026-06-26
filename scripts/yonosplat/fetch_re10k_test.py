# Fetch a real RealEstate10K test scene (the model's own benchmark) and build the encoder input bins.
# Downloads one pixelSplat-format .torch chunk from HF (lhmd/re10k_torch), finds a scene that's in
# YoNoSplat's evaluation index, extracts its 2 context views + (normalized) intrinsics, and writes
# re10k_image.bin [1,2,3,224,224] + re10k_intr.bin [1,2,3,3] (+ preview PNGs). Then:
#   adb push re10k_image.bin re10k_intr.bin /data/local/tmp/vxrt/yono/
#   adb shell ".../vx_yonosplat encoder_fp16.vxm re10k_image.bin re10k_intr.bin out.ppm --extr CAM"
import io, json, os, urllib.request
import numpy as np
import torch
from PIL import Image

OUT = "/tmp"
CHUNK = f"{OUT}/re10k_test/000000.torch"
URL = "https://huggingface.co/datasets/lhmd/re10k_torch/resolve/main/test/000000.torch"
EVIDX = "/tmp/YoNoSplat/assets/evaluation_index_re10k.json"  # ships with the YoNoSplat repo

os.makedirs(f"{OUT}/re10k_test", exist_ok=True)
if not os.path.exists(CHUNK):
    print("downloading", URL, "(~106MB) ...")
    urllib.request.urlretrieve(URL, CHUNK)

chunk = torch.load(CHUNK, map_location="cpu", weights_only=False)
evidx = json.load(open(EVIDX))
scene = next((s for s in chunk if s["key"] in evidx and evidx[s["key"]]), None)
assert scene is not None, "no chunk scene found in the eval index"
ev = evidx[scene["key"]]
ctx, tgt = ev["context"], ev["target"]
print("scene", scene["key"], "context", ctx, "target", tgt, "overlap", round(ev.get("overlap", 0), 3))


def decode(i):
    return Image.open(io.BytesIO(scene["images"][i].numpy().tobytes())).convert("RGB")


imgs, Ks = [], []
for fi in ctx:
    im = decode(fi).resize((224, 224), Image.BILINEAR)
    imgs.append(np.transpose(np.asarray(im, np.float32) / 255.0, (2, 0, 1)))  # CHW, [0,1] RGB
    c = scene["cameras"][fi].numpy()  # [18] = fx,fy,cx,cy,0,0, w2c(3x4); intrinsics are normalized
    Ks.append(np.array([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1]], np.float32))

np.stack(imgs)[None].astype(np.float32).tofile(f"{OUT}/re10k_image.bin")  # [1,2,3,224,224]
np.stack(Ks)[None].astype(np.float32).tofile(f"{OUT}/re10k_intr.bin")     # [1,2,3,3]
Image.fromarray((np.concatenate([i.transpose(1, 2, 0) for i in imgs], 1) * 255).astype(np.uint8)).save(f"{OUT}/re10k_ctx.png")
decode(tgt[len(tgt) // 2]).resize((224, 224)).save(f"{OUT}/re10k_target.png")
print(f"wrote {OUT}/re10k_image.bin [1,2,3,224,224] + {OUT}/re10k_intr.bin [1,2,3,3] (+ ctx/target pngs)")
