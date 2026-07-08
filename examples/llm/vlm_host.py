#!/usr/bin/env python3
"""Host front-end for vknn_vlm: tokenizes prompts, preprocesses images, streams the reply.

vknn_vlm runs on the device and consumes token ids + a raw fp32 pixel tile; this script owns the
tokenizer (HF, GPT2-style BPE) and the image preprocessing, mirroring the HF processor with image
splitting off: resize to exactly IMGxIMG (aspect distortion is the processor's behavior), rescale
1/255, normalize mean=std=0.5, CHW fp32. Each image is pushed once per turn and spliced on-device.

  vlm_host.py --serial <SERIAL> --tokenizer <dir with tokenizer.json + chat template>
              [--image photo.jpg] [--question "..."] [--ddir /data/local/tmp/vknn/smolvlm2]
              [--model smolvlm2-2.2b-fp16.vxm] [--max-tokens N] [--temp T] [--once]

With --image/--question the turn runs non-interactively (--once exits after it); otherwise each
stdin line is a question, and `/img <path>` loads a new image for the following questions.
"""
import argparse
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from transformers import AutoProcessor

IMG = 384


def preprocess(path):
    # LANCZOS mirrors the HF Idefics3 processor's resample default (preprocessor_config "resample": 1).
    im = Image.open(path).convert("RGB").resize((IMG, IMG), Image.LANCZOS)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]  # [1,3,IMG,IMG]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--ddir", default="/data/local/tmp/vknn/smolvlm2")
    ap.add_argument("--model", default="smolvlm2-2.2b-fp16.vxm")
    ap.add_argument("--tokenizer", required=True, help="local dir with the processor/tokenizer files")
    ap.add_argument("--precision", default="low")
    ap.add_argument("--image", default="")
    ap.add_argument("--question", default="")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.tokenizer)
    tok = proc.tokenizer
    eos = tok.convert_tokens_to_ids("<end_of_utterance>")
    image_token = tok.convert_tokens_to_ids("<image>")

    dev_cmd = (f"cd {args.ddir} && ./vknn_vlm {args.model} --backend vulkan "
               f"--precision {args.precision} --max-tokens {args.max_tokens} "
               f"--temp {args.temp} --top-k {args.top_k} --top-p {args.top_p} "
               f"--eos {eos} --image-token {image_token}")
    dev = subprocess.Popen(["adb", "-s", args.serial, "shell", dev_cmd],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
                           text=True, bufsize=1)

    def push_image(path):
        px = preprocess(path)
        local = "/tmp/vlm_pixels.bin"
        px.tofile(local)
        remote = f"{args.ddir}/pixels.bin"
        subprocess.run(["adb", "-s", args.serial, "push", local, remote],
                       check=True, capture_output=True)
        dev.stdin.write(f"i {remote}\n")
        dev.stdin.flush()
        line = dev.stdout.readline().strip()
        if line != "IMG_OK":
            print(f"image load failed: {line}", file=sys.stderr)
            return False
        return True

    def ask(question, with_image):
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": question}]
        text = proc.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True)
        if with_image:
            # The processor expands <image> to the per-tile token run; reproduce it through the
            # processor itself so the ids match the device splice exactly.
            ids = proc(text=text, images=[Image.new("RGB", (IMG, IMG))], return_tensors="np",
                       images_kwargs={"do_image_splitting": False})["input_ids"][0]
        else:
            ids = tok.encode(text)
        dev.stdin.write(" ".join(str(int(i)) for i in ids) + "\n")
        dev.stdin.flush()
        gen, prev, t0, tfirst = [], "", time.time(), None
        while True:
            line = dev.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if line == "END":
                break
            if line == "ERR":
                return None
            gen.append(int(line))
            if tfirst is None:
                tfirst = time.time()
            text_now = tok.decode(gen)
            sys.stdout.write(text_now[len(prev):])
            sys.stdout.flush()
            prev = text_now
        dt = time.time() - (tfirst or t0)
        n = len(gen)
        ttft = (tfirst or time.time()) - t0
        print(f"\n[{n} tokens, TTFT {ttft:.2f}s, {((n - 1) / dt) if dt > 0 and n > 1 else 0:.1f} tok/s]")
        return gen

    have_image = False
    if args.image:
        have_image = push_image(args.image)
        if not have_image:
            return 1
    if args.question:
        ask(args.question, have_image)
        have_image = False  # the image joined that turn; the device consumed its rows
        if args.once:
            dev.stdin.close()
            return 0

    print("questions on stdin; '/img <path>' loads a new image. Ctrl-D quits.", file=sys.stderr)
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("/img "):
            have_image = push_image(raw[5:].strip())
            continue
        ask(raw, have_image)
        have_image = False  # an image joins exactly one turn (mirrors vknn_vlm)
    dev.stdin.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
