#!/usr/bin/env python3
"""Download the YoNoSplat encoder artifacts from HuggingFace.

The model files are too large for the git repo, so they live in a HuggingFace model repo. This pulls
everything (the model + the sample inputs/goldens the example config uses) into one folder.

  python benchmark/fetch_model.py [--repo katolikov/yonosplat-vknn] [--out benchmark/models] [--files ...]

Contents of the repo:
  yonosplat_encoder.onnx   8-view encoder graph (external weights in weights.bin)
  weights.bin              fp32 weights for the ONNX (~3.6 GB)
  encoder8_fp16.vxm        compiled fp16 vknn model, ready to run (~2.9 GB)
  image8.npy, intr8.npy    sample real 8-frame input
  *_gold.npy               onnxruntime goldens for the six outputs
Then run:  python benchmark/run.py run benchmark/yonosplat.json

A private repo needs a token: `hf auth login`, or pass --token, or set HF_TOKEN.
"""
import argparse, os
from huggingface_hub import hf_hub_download, snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="katolikov/yonosplat-vknn")
ap.add_argument("--out", default="benchmark/models")
ap.add_argument("--files", nargs="*", default=None, help="specific files; default = whole repo")
ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
if args.files:
    for f in args.files:
        p = hf_hub_download(repo_id=args.repo, filename=f, repo_type="model", local_dir=args.out, token=args.token)
        print(f"  {f}  -> {p}")
else:
    snapshot_download(repo_id=args.repo, repo_type="model", local_dir=args.out, token=args.token)
    print("  downloaded all files")
print(f"\ndone -> {args.out}.  run:  python benchmark/run.py run benchmark/yonosplat.json")
