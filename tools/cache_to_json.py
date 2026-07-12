#!/usr/bin/env python3
"""Convert a VKNN warm-start model cache (`*.cache`, MessagePack) to readable JSON.

The engine writes one `.cache` beside each model (see src/core/cache_codec.cpp): a MessagePack map
holding the device identity (vendor/device/driver, pipeline-cache UUID), the kernel hash, and an array
of per-variant entries. Each variant carries its fusion flags, the winograd/conv autotune selections
(`tune` / `tunelvl`), an opaque Vulkan `pipeline` cache blob, and a `weights` map of prepacked weights
(raw little-endian float bytes).

Binary fields are large and opaque, so by default they are summarized (byte length + sha256) and weight
blobs are shown as a float count. Flags widen that:

    cache_to_json.py model.cache                 # -> stdout, blobs summarized
    cache_to_json.py model.cache -o model.json   # -> file
    cache_to_json.py model.cache --base64        # include pipeline/UUID bytes as base64
    cache_to_json.py model.cache --weights-floats # decode weight blobs to float arrays (large)

No third-party dependency: the small MessagePack subset the codec emits is decoded inline.
"""
import argparse
import base64
import hashlib
import json
import struct
import sys


class _Unpacker:
    """Minimal MessagePack decoder covering exactly what cache_codec.cpp emits."""

    def __init__(self, data: bytes):
        self.d = data
        self.i = 0

    def _raw(self, n: int) -> bytes:
        v = self.d[self.i:self.i + n]
        if len(v) != n:
            raise ValueError("truncated MessagePack stream")
        self.i += n
        return v

    def _uint(self, n: int) -> int:
        return int.from_bytes(self._raw(n), "big")

    def _int(self, n: int) -> int:
        return int.from_bytes(self._raw(n), "big", signed=True)

    def _str(self, n: int) -> str:
        return self._raw(n).decode("utf-8", "replace")

    def _array(self, n: int) -> list:
        return [self.read() for _ in range(n)]

    def _map(self, n: int) -> dict:
        return {self.read(): self.read() for _ in range(n)}

    def read(self):
        c = self.d[self.i]
        self.i += 1
        if c <= 0x7F:
            return c                      # positive fixint
        if c >= 0xE0:
            return c - 0x100             # negative fixint
        if 0x80 <= c <= 0x8F:
            return self._map(c & 0x0F)   # fixmap
        if 0x90 <= c <= 0x9F:
            return self._array(c & 0x0F)  # fixarray
        if 0xA0 <= c <= 0xBF:
            return self._str(c & 0x1F)   # fixstr
        return self._read_ext(c)

    def _read_ext(self, c: int):
        if c == 0xC0:
            return None
        if c == 0xC2:
            return False
        if c == 0xC3:
            return True
        if c == 0xC4:
            return self._raw(self._uint(1))   # bin8  -> bytes
        if c == 0xC5:
            return self._raw(self._uint(2))   # bin16
        if c == 0xC6:
            return self._raw(self._uint(4))   # bin32
        if c == 0xCA:
            return struct.unpack(">f", self._raw(4))[0]
        if c == 0xCB:
            return struct.unpack(">d", self._raw(8))[0]
        if c in (0xCC, 0xCD, 0xCE, 0xCF):
            return self._uint(1 << (c - 0xCC))  # uint 8/16/32/64
        if c in (0xD0, 0xD1, 0xD2, 0xD3):
            return self._int(1 << (c - 0xD0))   # int 8/16/32/64
        if c == 0xD9:
            return self._str(self._uint(1))     # str8
        if c == 0xDA:
            return self._str(self._uint(2))     # str16
        if c == 0xDB:
            return self._str(self._uint(4))     # str32
        if c == 0xDC:
            return self._array(self._uint(2))   # array16
        if c == 0xDD:
            return self._array(self._uint(4))   # array32
        if c == 0xDE:
            return self._map(self._uint(2))     # map16
        if c == 0xDF:
            return self._map(self._uint(4))     # map32
        raise ValueError(f"unsupported MessagePack byte 0x{c:02x} at offset {self.i - 1}")


def _jsonify(obj, key, parent_key, args):
    if isinstance(obj, dict):
        return {k: _jsonify(v, k, key, args) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v, key, parent_key, args) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return _jsonify_bytes(bytes(obj), key, parent_key, args)
    return obj  # int, float, str, bool, None


def _jsonify_bytes(b, key, parent_key, args):
    if key == "pipelineCacheUUID":
        return b.hex()
    if parent_key == "weights":  # raw little-endian float32 vector
        count = len(b) // 4
        if args.weights_floats:
            return list(struct.unpack("<%df" % count, b[:count * 4]))
        return {"floats": count}
    out = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    if args.base64:
        out["base64"] = base64.b64encode(b).decode("ascii")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert a VKNN .cache (MessagePack) to JSON.")
    ap.add_argument("cache", help="path to the .cache file")
    ap.add_argument("-o", "--output", help="write JSON here (default: stdout)")
    ap.add_argument("--base64", action="store_true", help="include opaque blobs (pipeline) as base64")
    ap.add_argument("--weights-floats", action="store_true", help="decode weight blobs to float arrays (large)")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2)")
    args = ap.parse_args(argv)

    with open(args.cache, "rb") as f:
        data = f.read()
    try:
        doc = _Unpacker(data).read()
    except (ValueError, IndexError) as e:
        sys.exit(f"error: {args.cache} is not a valid VKNN cache ({e})")
    if not isinstance(doc, dict):  # the cache root is always a MessagePack map (cache_codec.cpp)
        sys.exit(f"error: {args.cache} is not a VKNN cache (root is {type(doc).__name__}, expected a map)")

    result = _jsonify(doc, None, None, args)
    text = json.dumps(result, indent=args.indent, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
