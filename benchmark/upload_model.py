#!/usr/bin/env python3
"""Create the HuggingFace model repo and upload the YoNoSplat encoder artifacts.

Needs a WRITE token: `hf auth login`, or pass --token, or set HF_TOKEN. One-time publishing step;
end users only ever run fetch_model.py.

  python benchmark/upload_model.py \
      --repo katolikov/yonosplat-vknn --private \
      --onnx /tmp/YoNoSplat/onnx/yonosplat_encoder.onnx \
      --weights /tmp/YoNoSplat/onnx/weights.bin \
      --vxm /tmp/encoder8_fp16.vxm
"""
import argparse, os
from huggingface_hub import HfApi, create_repo

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="katolikov/yonosplat-vknn")
ap.add_argument("--private", action="store_true")
ap.add_argument("--onnx", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--vxm", required=True)
ap.add_argument("--card", default=os.path.join(os.path.dirname(__file__), "hf_README.md"))
ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
args = ap.parse_args()

api = HfApi(token=args.token)
create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True, token=args.token)
print(f"repo: https://huggingface.co/{args.repo} (private={args.private})")

for local, remote in [(args.card, "README.md"), (args.onnx, "yonosplat_encoder.onnx"),
                      (args.weights, "weights.bin"), (args.vxm, "encoder8_fp16.vxm")]:
    sz = os.path.getsize(local) / 1e9
    print(f"uploading {remote} ({sz:.2f} GB) ...")
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=args.repo, repo_type="model", token=args.token)
print("done.")
