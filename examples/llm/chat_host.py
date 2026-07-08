#!/usr/bin/env python3
"""Terminal chat front-end for the on-device VKNN GPU decoder (vknn_chat).

Tokenization is the one host dependency: this drives the device binary over adb, feeding prompt
token ids on its stdin and reading generated ids from its stdout, detokenizing + streaming them to
the terminal (typewriter effect) with a live tokens/s counter. All model compute stays on the GPU
in the device binary; this process only tokenizes, displays, and holds the REPL. The conversation
KV cache lives in the device binary and persists across turns.

Usage:
  chat_host.py --serial <device-serial> --ddir /data/local/tmp/vknn/qwen --model qwen_chat.vxm \
               --tokenizer qwen-onnx --precision low --fp32-tensors CSV --max-tokens 128 [--temp 0]
"""
import argparse
import subprocess
import sys
import time
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--ddir", default="/data/local/tmp/vknn/qwen")
    ap.add_argument("--model", default="qwen_chat.vxm", help="model filename inside --ddir on device")
    ap.add_argument("--tokenizer", default="qwen-onnx", help="local dir with tokenizer.json etc.")
    ap.add_argument("--precision", default="low")
    ap.add_argument("--fp32-tensors", default="")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 151643

    dev_cmd = (f"cd {args.ddir} && ./vknn_chat {args.model} --backend vulkan "
               f"--precision {args.precision} --max-tokens {args.max_tokens} "
               f"--temp {args.temp} --top-k {args.top_k} --top-p {args.top_p} --eos {eos}")
    if args.fp32_tensors:
        dev_cmd += f" --fp32-tensors {args.fp32_tensors}"
    proc = subprocess.Popen(["adb", "-s", args.serial, "shell", dev_cmd],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
                            text=True, bufsize=1)

    print(f"vknn_chat on {args.serial} — completion model; type a prompt, Ctrl-D to quit.\n")
    try:
        while True:
            try:
                user = input("\x1b[1myou>\x1b[0m ")
            except EOFError:
                break
            if not user.strip():
                continue
            ids = tok.encode(user)
            proc.stdin.write(" ".join(str(i) for i in ids) + "\n")
            proc.stdin.flush()

            gen, prev, t0 = [], "", None
            sys.stdout.write("\x1b[36m")  # cyan for the completion
            sys.stdout.flush()
            while True:
                line = proc.stdout.readline()
                if not line:
                    print("\n[device stream closed]")
                    return
                line = line.rstrip("\n")
                if line == "END":
                    break
                try:
                    gen.append(int(line))
                except ValueError:
                    continue
                if t0 is None:
                    t0 = time.time()
                text = tok.decode(gen)
                sys.stdout.write(text[len(prev):])
                sys.stdout.flush()
                prev = text
            dt = (time.time() - t0) if t0 else 0.0
            rate = (len(gen) / dt) if dt > 0 else 0.0
            sys.stdout.write(f"\x1b[0m\n\x1b[2m[{len(gen)} tokens, {rate:.1f} tok/s]\x1b[0m\n\n")
            sys.stdout.flush()
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.terminate()


if __name__ == "__main__":
    main()
