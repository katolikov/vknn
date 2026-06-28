#!/usr/bin/env python3
"""Create the HuggingFace model repo and upload the YoNoSplat encoder artifacts + sample I/O.

Needs a WRITE token: `hf auth login`, or pass --token, or set HF_TOKEN. One-time publishing step;
end users only ever run fetch_model.py.

  python benchmark/scripts/upload_model.py --repo katolikov/yonosplat-vknn [--private] \
      --onnx /tmp/YoNoSplat/onnx/yonosplat_encoder.onnx \
      --weights /tmp/YoNoSplat/onnx/weights.bin \
      --vxm /tmp/encoder8_fp16.vxm \
      --samples /tmp/yono_val          # dir of image8.npy / intr8.npy / *_gold.npy (optional)
"""
import argparse, glob, os
from huggingface_hub import HfApi, create_repo

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="katolikov/yonosplat-vknn")
parser.add_argument("--private", action="store_true")
parser.add_argument("--onnx", required=True)
parser.add_argument("--weights", required=True)
parser.add_argument("--vxm", required=True)
parser.add_argument("--samples", default=None, help="dir whose *.npy are uploaded at the repo root")
parser.add_argument("--card", default=os.path.join(os.path.dirname(__file__), "hf_README.md"))
parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
args = parser.parse_args()

api = HfApi(token=args.token)
create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True, token=args.token)
print(f"repo: https://huggingface.co/{args.repo} (private={args.private})")

uploads = [(args.card, "README.md"), (args.onnx, "yonosplat_encoder.onnx"),
           (args.weights, "weights.bin"), (args.vxm, "encoder8_fp16.vxm")]
if args.samples:
    for p in sorted(glob.glob(os.path.join(args.samples, "*.npy"))):
        uploads.append((p, os.path.basename(p)))

for local, remote in uploads:
    sz = os.path.getsize(local) / 1e9
    print(f"uploading {remote} ({sz:.2f} GB) ...", flush=True)
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=args.repo, repo_type="model", token=args.token)
print("done.")
