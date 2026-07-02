#!/usr/bin/env python3
"""Rebuild a RUNNABLE ONNX from an onnx_dump.py --json dump (when the original .onnx is gone).

The dump records every node (op_type, inputs, outputs, attributes), every tensor's dtype + a
(shape-inferred) shape, and the graph input/output dtypes+shapes -- but NOT the numeric contents of
initializers or Constant nodes (a Constant's `value` attr is dumped as the string "<tensor DTYPE
[shape]>", which still tells us that constant's OWN dtype+shape). This tool reconstructs an
equivalent, runnable graph that ONNX Runtime executes end-to-end and from which golden I/O is dumped.

Two problems make the dump non-trivial to replay:

  1. SSA is violated: the trace reuses tensor names (e.g. two Cast nodes both emit "Cast_output_0";
     6 different nodes emit "Mul_3_output_0"). We SSA-rename -- every node output becomes unique and
     each input binds to the nearest PRECEDING producer of that name (a graph input / initializer
     otherwise). Declared graph outputs are re-exposed by Identity to their original names. Because
     names collapse, the dump's per-tensor `shape` is last-writer-wins and is only trustworthy for
     uniquely-named tensors -- so we do NOT rely on it; we recompute every shape ourselves.

  2. The numeric contents of the ~347 Constant nodes are unknown, yet ~191 of them are INT64 shape
     vectors / Slice bounds / axes / Gather indices that must be self-consistent or Reshape / Resize /
     Expand / ConstantOfShape fail at runtime. We recover them with a VALUE INTERPRETER: walking the
     trace in order, we compute a concrete value (and shape) for every shape/index tensor, seeded from
     the fixed input shapes. `Shape(X)` returns X's recomputed concrete shape, so Slice/Concat/Gather/
     Unsqueeze over shapes yield correct size vectors; every Reshape/Resize/Expand target is therefore
     materialised as a concrete, self-consistent initializer. Constants are synthesized by role:
       * INT64 shape-control leaves (Slice starts/ends/axes/steps, Gather idx, Unsqueeze/Squeeze axes,
         Concat baked dims, Reshape/Expand/Resize/ConstantOfShape targets) get values that make the
         geometry valid; where a leaf value is genuinely free we pick the neutral one.
       * big FLOAT axis vectors are coordinate aranges for the
         grid-sample warps -> arange(N).
       * remaining FLOAT/FLOAT16 operands (Mul/Add/Sub/Div/Clip/Pow free operands) only affect numeric
         output, not shape -> neutral values (1 for Mul/Div, 0 for Add/Sub, small for the rest).
     Weights (all FLOAT16 initializers) get small random values.

The result is a faithful repro of the op / dtype / dynamic-shape STRUCTURE (the permanent engine
regression artifact) -- not the trained weights. Validate by running it in ORT (this script's
--check) and in vknn, and by diffing vknn inferShapes against ORT's shapes.

Usage:
  tools/rebuild_onnx_from_dump.py model.json -o rebuilt.onnx \
      [--inputs-dir DIR] [--goldens-dir DIR] [--seed N] [--check]
Requires: onnx, numpy (onnxruntime only for --check).
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ELEM = {
    "FLOAT": TensorProto.FLOAT, "FLOAT16": TensorProto.FLOAT16, "DOUBLE": TensorProto.DOUBLE,
    "INT64": TensorProto.INT64, "INT32": TensorProto.INT32, "INT8": TensorProto.INT8,
    "UINT8": TensorProto.UINT8, "BOOL": TensorProto.BOOL,
}
NP_OF = {
    "FLOAT": np.float32, "FLOAT16": np.float16, "DOUBLE": np.float64,
    "INT64": np.int64, "INT32": np.int32, "INT8": np.int8, "UINT8": np.uint8, "BOOL": np.bool_,
}
# onnx `to`/elem_type int -> numpy dtype (for Cast + Constant value dtype from int)
ELEM_INT_TO_NP = {
    1: np.float32, 2: np.uint8, 3: np.int8, 6: np.int32, 7: np.int64,
    9: np.bool_, 10: np.float16, 11: np.float64,
}


def concrete(shape):
    return shape is not None and all(isinstance(x, int) for x in shape)


def _const_meta(node):
    """(dtype_name, shape_list) for a Constant node whose value attr is '<tensor DTYPE [..]>'."""
    v = str(node["attributes"].get("value", ""))
    m = re.search(r"<tensor (\w+) \[([\d,\s]*)\]>", v)
    if not m:
        return None, None
    shp = [int(x) for x in m.group(2).split(",") if x.strip()]
    return m.group(1), shp


# ------------------------------------------------------------------ geometry helpers
def _bcast(a, b):
    """Numpy broadcast of two concrete shape lists (or None if either unknown)."""
    if a is None or b is None:
        return None
    ra, rb = list(a), list(b)
    n = max(len(ra), len(rb))
    ra = [1] * (n - len(ra)) + ra
    rb = [1] * (n - len(rb)) + rb
    out = []
    for x, y in zip(ra, rb):
        if not (isinstance(x, int) and isinstance(y, int)):
            return None
        if x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        elif x == y:
            out.append(x)
        else:
            out.append(max(x, y))
    return out


def _broadcastable(src, tgt):
    """True if `src` (a concrete shape) can Expand to `tgt` (a concrete shape)."""
    if not (concrete(src) and concrete(tgt)):
        return False
    rs, rt = list(src), list(tgt)
    n = max(len(rs), len(rt))
    rs = [1] * (n - len(rs)) + rs
    rt = [1] * (n - len(rt)) + rt
    return all(s == t or s == 1 for s, t in zip(rs, rt))


def _reshape_valid(tgt, dsh):
    """True if reshape target `tgt` (with -1/0 semantics) is element-count-compatible with data
    shape `dsh` (both concrete lists)."""
    total = int(np.prod(dsh)) if dsh else 1
    known, minus = 1, 0
    for k, t in enumerate(tgt):
        t = int(t)
        if t == -1:
            minus += 1
        elif t == 0:
            known *= dsh[k] if k < len(dsh) else 1
        else:
            known *= t
    if minus > 1 or known <= 0:
        return False
    return (total % known == 0) if minus else (total == known)


def _conv_out(ish, wsh, attrs, transpose):
    """NCHW conv / conv-transpose spatial geometry -> output shape (concrete list) or None."""
    if not concrete(ish) or len(ish) < 3:
        return None
    n = ish[0]
    spatial = ish[2:]
    ks = attrs.get("kernel_shape")
    if ks is None and concrete(wsh):
        ks = list(wsh[2:])
    if ks is None:
        return None
    nsp = len(spatial)
    strides = attrs.get("strides") or [1] * nsp
    dil = attrs.get("dilations") or [1] * nsp
    pads = attrs.get("pads") or [0] * (2 * nsp)
    out_pad = attrs.get("output_padding") or [0] * nsp
    group = attrs.get("group", 1)
    if len(ks) != nsp or len(strides) != nsp or len(dil) != nsp or len(pads) != 2 * nsp:
        return None
    if transpose:
        # Cout from weight [Cin, Cout/group, kH, kW]
        cout = wsh[1] * group if concrete(wsh) else None
        osp = attrs.get("output_shape")
        if osp is not None:
            spatial_out = list(osp)
        else:
            spatial_out = []
            for i, s in enumerate(spatial):
                p = pads[i] + pads[i + len(spatial)]
                o = strides[i] * (s - 1) + out_pad[i] + ((ks[i] - 1) * dil[i] + 1) - p
                spatial_out.append(o)
    else:
        cout = wsh[0] if concrete(wsh) else None
        spatial_out = []
        for i, s in enumerate(spatial):
            p = pads[i] + pads[i + len(spatial)]
            o = (s + p - (dil[i] * (ks[i] - 1) + 1)) // strides[i] + 1
            spatial_out.append(o)
    if cout is None:
        return None
    return [n, cout] + spatial_out


def _norm_axis(a, rank):
    return a + rank if a < 0 else a


class Interp:
    """Value/shape interpreter over the SSA-renamed trace.

    val[u]  : concrete numpy array for a fully-known (shape/index) tensor, else absent
    shp[u]  : concrete python int-list shape, else None
    """

    def __init__(self, rng):
        self.val = {}
        self.shp = {}
        self.rng = rng

    def gv(self, u):
        return self.val.get(u)

    def gs(self, u):
        return self.shp.get(u)

    # ---- Constant synthesis (the crux) ----------------------------------------
    def make_const(self, uo, dtn, shp, roleset):
        """Return a numpy value for Constant `uo` from its ROLE only (dtype dtn, value-attr shape shp).

        Consumer-shape-dependent values (Reshape/Resize/Expand/ConstantOfShape targets, Slice bounds)
        are patched later by `_fix_shape_control` right before the consumer runs, when the data-input
        shape is known. Here we produce a valid provisional value.
        """
        npdt = NP_OF[dtn]
        n = int(np.prod(shp)) if shp else 1

        if dtn == "INT64":
            return self._int64_const(shp, n, roleset).astype(np.int64)

        if dtn in ("FLOAT", "FLOAT16", "DOUBLE"):
            # big 1-D vectors that feed Mul in a grid-warp block are coordinate aranges. Normalise to
            # [0,1) so they stay bounded -- an un-normalised full-width pixel ramp multiplied through the deep
            # color/warp chain overflows fp16 (-> inf/nan -> zeroed outputs).
            if len(shp) == 1 and shp[0] >= 16:
                return (np.arange(shp[0], dtype=np.float64) / shp[0]).astype(npdt)
            # Operand values are kept benign for shape/finiteness but NON-DEGENERATE so signal does not
            # cancel to 0 (a real regression vehicle). Default ~1.0 keeps Div/Reciprocal/Pow finite;
            # additive operands get a small non-zero offset instead of exact 0 (avoids x-x=0 collapse);
            # comparison thresholds are near 0 so masks are mixed, not all-false. Clip stays wide but
            # finite (fp16-safe, not 1e4 which can overflow when multiplied downstream).
            val = np.full(n, 1.0, dtype=np.float64)
            if any(op == "Clip" and pos == 1 for (op, pos) in roleset):
                val[:] = -1.0 # clip min: |window| <= 1 so products of clipped tensors never amplify
            elif any(op == "Clip" and pos == 2 for (op, pos) in roleset):
                val[:] = 1.0  # clip max (a wider range compounds through row-x-col Mul chains to fp16 inf)
            elif any(op in ("Greater", "GreaterOrEqual") for (op, _) in roleset):
                val[:] = 0.0                                              # threshold at 0 -> mixed mask
            elif any(op == "Div" and pos == 1 for (op, pos) in roleset):
                val[:] = 16.0                                             # divisor -> shrinks magnitude (fp16-safe)
            elif any(op == "Mul" for (op, _) in roleset):
                val[:] = 0.0625                                           # dampen products: tensor-x-tensor Muls of
                                                                          # uint8-scale planes stack magnitude fast
            elif any(op == "Pow" for (op, _) in roleset):
                val[:] = 1.0                                              # exponent 1 -> identity (no blow-up)
            elif roleset and all(op in ("Add", "Sub") for (op, _) in roleset):
                val[:] = 0.1                                              # small non-zero additive term
            return val.reshape(shp).astype(npdt) if shp else np.array(val[0], dtype=npdt)

        if dtn == "BOOL":
            return (np.zeros(n, dtype=np.bool_).reshape(shp) if shp else np.array(False))
        return (np.zeros(n, dtype=npdt).reshape(shp) if shp else np.zeros((), dtype=npdt))

    def _int64_const(self, shp, n, roleset):
        """Provisional INT64 constant from role only (patched later where shape-dependent)."""
        # Slice bounds: neutral full-range so any Slice is valid until patched.
        if any(op == "Slice" and pos == 1 for (op, pos) in roleset):
            return np.zeros(n, dtype=np.int64)                              # starts -> 0
        if any(op == "Slice" and pos == 2 for (op, pos) in roleset):
            return np.full(n, np.iinfo(np.int64).max, dtype=np.int64)       # ends -> +inf
        if any(op == "Slice" and pos == 3 for (op, pos) in roleset):
            return np.arange(n, dtype=np.int64)                            # axes -> 0..n-1
        if any(op == "Slice" and pos == 4 for (op, pos) in roleset):
            return np.ones(n, dtype=np.int64)                              # steps -> 1
        # Gather index -> 0 (valid for axis length >= 1)
        if any(op == "Gather" and pos == 1 for (op, pos) in roleset):
            return np.zeros(n, dtype=np.int64)
        # Unsqueeze / Squeeze / Reduce axes -> distinct valid axes 0..n-1
        if any(op in ("Unsqueeze", "Squeeze", "ReduceSum") and pos == 1 for (op, pos) in roleset):
            return np.arange(n, dtype=np.int64)
        # Concat baked dims / Reshape / Expand / ConstantOfShape targets -> positive placeholder;
        # patched to the geometric target by _fix_shape_control before the consumer runs.
        return np.ones(n, dtype=np.int64)

    # ---- op execution / shape propagation -------------------------------------
    def run(self, op, ins, outs, attrs):
        """Compute val/shp for a node's outputs. ins/outs are unique names. Returns nothing."""
        getv = [self.gv(u) if u else None for u in ins]
        gets = [self.gs(u) if u else None for u in ins]
        o0 = outs[0] if outs else None

        def setv(u, arr):
            arr = np.asarray(arr)
            self.val[u] = arr
            self.shp[u] = list(arr.shape)

        def sets(u, s):
            self.shp[u] = list(s) if s is not None else None

        known = lambda: all(v is not None for v in getv[:_nin(op)])

        if op == "Shape":
            s = gets[0]
            if concrete(s):
                start = attrs.get("start", 0)
                end = attrs.get("end", len(s))
                start = _norm_axis(start, len(s)); end = _norm_axis(end, len(s)) if end < 0 else min(end, len(s))
                setv(o0, np.array(s[start:end], dtype=np.int64))
            else:
                sets(o0, None)
            return

        if op == "Constant":
            return  # value already assigned before run()

        if op == "Cast":
            to = attrs.get("to")
            npdt = ELEM_INT_TO_NP.get(to, np.float32)
            if getv[0] is not None:
                setv(o0, getv[0].astype(npdt))
            else:
                sets(o0, gets[0])
            return

        if op in ("Slice",):
            data = getv[0]
            if data is not None and getv[1] is not None and getv[2] is not None:
                starts = np.atleast_1d(getv[1]).astype(np.int64)
                ends = np.atleast_1d(getv[2]).astype(np.int64)
                axes = (np.atleast_1d(getv[3]).astype(np.int64)
                        if len(getv) > 3 and getv[3] is not None else np.arange(len(starts)))
                steps = (np.atleast_1d(getv[4]).astype(np.int64)
                         if len(getv) > 4 and getv[4] is not None else np.ones(len(starts), np.int64))
                out = data
                sl = [slice(None)] * data.ndim
                for st, en, ax, sp in zip(starts, ends, axes, steps):
                    ax = int(ax) % data.ndim
                    en = min(int(en), data.shape[ax]) if en < np.iinfo(np.int64).max else data.shape[ax]
                    sl[ax] = slice(int(st), en, int(sp))
                setv(o0, out[tuple(sl)])
            else:
                # data shape known but non-const: compute output shape from starts/ends/axes
                dsh = gets[0]
                if concrete(dsh) and getv[1] is not None and getv[2] is not None:
                    starts = np.atleast_1d(getv[1]).astype(np.int64)
                    ends = np.atleast_1d(getv[2]).astype(np.int64)
                    axes = (np.atleast_1d(getv[3]).astype(np.int64)
                            if len(getv) > 3 and getv[3] is not None else np.arange(len(starts)))
                    steps = (np.atleast_1d(getv[4]).astype(np.int64)
                             if len(getv) > 4 and getv[4] is not None else np.ones(len(starts), np.int64))
                    osh = list(dsh)
                    for st, en, ax, sp in zip(starts, ends, axes, steps):
                        ax = int(ax) % len(osh)
                        e = min(int(en), osh[ax]) if en < np.iinfo(np.int64).max else osh[ax]
                        s0 = int(st) if st >= 0 else osh[ax] + int(st)
                        osh[ax] = max(0, (e - s0 + int(sp) - 1) // int(sp))
                    sets(o0, osh)
                else:
                    sets(o0, None)
            return

        if op == "Concat":
            axis = attrs.get("axis", 0)
            vv = [v for v in getv if v is not None]
            if len(vv) == len(getv) and getv:
                setv(o0, np.concatenate([np.atleast_1d(v) for v in getv], axis=axis))
            else:
                # shape-only: sum sizes along `axis`, other dims from any concrete input
                shapes = [s for s in gets if concrete(s)]
                if shapes and all(concrete(s) for s in gets):
                    ax = axis % len(shapes[0])
                    out = list(shapes[0])
                    out[ax] = sum(s[ax] for s in gets)
                    sets(o0, out)
                else:
                    sets(o0, None)
            return

        if op == "Gather":
            axis = attrs.get("axis", 0)
            if getv[0] is not None and getv[1] is not None:
                setv(o0, np.take(getv[0], getv[1].astype(np.int64), axis=axis))
            elif concrete(gets[0]) and getv[1] is not None:
                # shape-only: result rank = data.rank - 1 + index.rank
                s = list(gets[0]); ax = axis % len(s)
                idx = np.atleast_1d(getv[1])
                irank = 0 if getv[1].ndim == 0 else idx.ndim
                out = s[:ax] + list(idx.shape if irank else []) + s[ax + 1:]
                sets(o0, out)
            else:
                sets(o0, None)
            return

        if op == "Unsqueeze":
            if getv[0] is not None and getv[1] is not None:
                out = getv[0]
                for ax in sorted(int(a) for a in np.atleast_1d(getv[1])):
                    out = np.expand_dims(out, ax)
                setv(o0, out)
            elif concrete(gets[0]) and getv[1] is not None:
                s = list(gets[0])
                r = len(s) + len(np.atleast_1d(getv[1]))
                for ax in sorted(int(a) % r for a in np.atleast_1d(getv[1])):
                    s.insert(ax, 1)
                sets(o0, s)
            else:
                sets(o0, None)
            return

        if op == "Squeeze":
            if getv[0] is not None:
                ax = tuple(int(a) for a in np.atleast_1d(getv[1])) if len(getv) > 1 and getv[1] is not None else None
                setv(o0, np.squeeze(getv[0], ax))
            elif concrete(gets[0]):
                s = list(gets[0])
                if len(getv) > 1 and getv[1] is not None:
                    for ax in sorted((int(a) % len(s) for a in np.atleast_1d(getv[1])), reverse=True):
                        if s[ax] == 1:
                            s.pop(ax)
                else:
                    s = [x for x in s if x != 1]
                sets(o0, s)
            else:
                sets(o0, None)
            return

        if op == "ConstantOfShape":
            if getv[0] is not None:
                fill = 0
                v = attrs.get("value")
                shp = tuple(int(x) for x in getv[0])
                setv(o0, np.zeros(shp, dtype=np.int64) + fill)
            else:
                sets(o0, None)
            return

        if op in ("Add", "Sub", "Mul", "Div", "Equal", "Where", "Min", "Max", "Pow", "Greater", "GreaterOrEqual", "And", "Or"):
            if op == "Where":
                if all(v is not None for v in getv[:3]):
                    setv(o0, np.where(getv[0], getv[1], getv[2]))
                    return
                sets(o0, _bcast(_bcast(gets[0], gets[1]), gets[2]))
                return
            if getv[0] is not None and getv[1] is not None:
                a, b = getv[0], getv[1]
                try:
                    if op == "Add":
                        r = a + b
                    elif op == "Sub":
                        r = a - b
                    elif op == "Mul":
                        r = a * b
                    elif op == "Div":
                        r = a / b if a.dtype.kind == "f" else a // b
                    elif op == "Equal":
                        r = np.equal(a, b)
                    elif op == "Min":
                        r = np.minimum(a, b)
                    elif op == "Max":
                        r = np.maximum(a, b)
                    elif op == "Pow":
                        r = np.power(a, b)
                    elif op == "Greater":
                        r = np.greater(a, b)
                    elif op == "GreaterOrEqual":
                        r = np.greater_equal(a, b)
                    else:
                        r = None
                    if r is not None:
                        setv(o0, r); return
                except Exception:
                    pass
            sets(o0, _bcast(gets[0], gets[1]))
            return

        if op in ("Reciprocal", "Neg", "Relu", "Sigmoid", "Sqrt", "Exp", "Abs", "Identity"):
            if getv[0] is not None and op in ("Reciprocal", "Neg", "Identity", "Abs"):
                if op == "Reciprocal":
                    setv(o0, 1.0 / getv[0])
                elif op == "Neg":
                    setv(o0, -getv[0])
                elif op == "Abs":
                    setv(o0, np.abs(getv[0]))
                else:
                    setv(o0, getv[0])
            else:
                sets(o0, gets[0])
            return

        if op == "Clip":
            sets(o0, gets[0]); return

        if op == "PRelu":
            sets(o0, gets[0]); return

        if op == "Reshape":
            if getv[0] is not None and getv[1] is not None:
                try:
                    setv(o0, getv[0].reshape([int(x) for x in getv[1]]))
                    return
                except Exception:
                    pass
            if getv[1] is not None and concrete(gets[0]):
                tgt = [int(x) for x in getv[1]]
                total = int(np.prod(gets[0]))
                known = int(np.prod([t for t in tgt if t > 0])) if any(t > 0 for t in tgt) else 1
                tgt = [t if t > 0 else (total // known if known else 0) for t in tgt]
                sets(o0, tgt)
            else:
                sets(o0, None)
            return

        if op == "Expand":
            if getv[1] is not None:
                tgt = [int(x) for x in getv[1]]
                sets(o0, _bcast(gets[0], tgt))
            else:
                sets(o0, None)
            return

        if op == "Transpose":
            perm = attrs.get("perm")
            if concrete(gets[0]):
                s = gets[0]
                perm = perm if perm is not None else list(range(len(s)))[::-1]
                sets(o0, [s[p] for p in perm])
            else:
                sets(o0, None)
            return

        if op in ("Conv", "ConvTranspose"):
            sets(o0, _conv_out(gets[0], gets[1], attrs, op == "ConvTranspose"))
            return

        if op == "Resize":
            # sizes input (index 3) wins; else scales (index 2)
            sizes = getv[3] if len(getv) > 3 and getv[3] is not None else None
            scales = getv[2] if len(getv) > 2 and getv[2] is not None else None
            if sizes is not None:
                sets(o0, [int(x) for x in sizes])
            elif scales is not None and concrete(gets[0]):
                sets(o0, [int(round(d * s)) for d, s in zip(gets[0], scales)])
            else:
                sets(o0, None)
            return

        if op == "GridSample":
            # out = [N, C_data, H_grid, W_grid]; grid is [N, H, W, 2]
            ds = gets[0]; gs = gets[1]
            if concrete(ds) and concrete(gs) and len(ds) == 4 and len(gs) == 4:
                sets(o0, [ds[0], ds[1], gs[1], gs[2]])
            else:
                sets(o0, None)
            return

        if op in ("ReduceMax", "ReduceMin", "ReduceSum", "ReduceMean"):
            axes = attrs.get("axes")
            if axes is None and len(getv) > 1 and getv[1] is not None:
                axes = [int(a) for a in np.atleast_1d(getv[1])]
            keep = attrs.get("keepdims", 1)
            if concrete(gets[0]):
                s = list(gets[0])
                if axes is None:
                    axes = list(range(len(s)))
                axes = [a % len(s) for a in axes]
                for a in sorted(axes, reverse=True):
                    if keep:
                        s[a] = 1
                    else:
                        s.pop(a)
                sets(o0, s)
            else:
                sets(o0, None)
            return

        # default: rank-preserving unary; copy input shape
        sets(o0, gets[0])
        return


# ops whose data input (pos 0) is a feature map -> eligible for rank-corrected binding
_FEATURE_CONSUMERS = {"Shape", "Resize", "Conv", "ConvTranspose", "GridSample", "Transpose",
                      "Relu", "PRelu", "Sigmoid"}


def _nin(op):
    return {"Where": 3, "Slice": 4, "Resize": 4, "GridSample": 2}.get(op, 2)


def _find_grid_flow(interp, grid_in, by_out):
    """Walk a grid tensor backward (Cast/Transpose) to the Add that fuses coord-grid + flow, and
    return the flow operand (the 4-D, small-channel conv output). Used to keep GridSample dependent
    on the convolutions after we rebuild its grid."""
    cur = grid_in
    for _ in range(6):
        n = by_out.get(cur)
        if n is None:
            return None
        if n.op_type in ("Cast", "Transpose"):
            cur = n.input[0]
            continue
        if n.op_type == "Add":
            # the grid = Add(coordinate_grid, flow); the flow is the operand that traces back to a
            # Conv/ConvTranspose (the optical flow), the coordinate grid traces to arange constants.
            # Prefer the operand that reaches a convolution -> keeps the outputs dependent on the whole
            # network. Fall back to the Cast-produced / non-builder operand.
            builder = ("Transpose", "Concat", "Expand", "Reshape", "Unsqueeze", "ConstantOfShape")
            conv_op, cast_op, other_op, any_op = None, None, None, None
            for inp in n.input:
                p = by_out.get(inp)
                if p is None:
                    continue
                if any_op is None:
                    any_op = inp
                if conv_op is None and _reaches_conv(inp, by_out, budget=4000):
                    conv_op = inp
                if p.op_type == "Cast" and cast_op is None:
                    cast_op = inp
                elif p.op_type not in builder and other_op is None:
                    other_op = inp
            return conv_op or cast_op or other_op or any_op
        return None
    return None


def _reaches_conv(name, by_out, budget=40):
    """Bounded backward search: does `name` transitively depend on a Conv/ConvTranspose?"""
    seen = set()
    stack = [name]
    while stack and budget > 0:
        budget -= 1
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        p = by_out.get(cur)
        if p is None:
            continue
        if p.op_type in ("Conv", "ConvTranspose"):
            return True
        for i in p.input:
            if i:
                stack.append(i)
    return False


def _norm_grid(interp, onnx_nodes, base, cn, tgt):
    """Squash a synthesized grid into GridSample's [-1,1] domain: g*(1/128)-1 maps a uint8-scale
    feature (0..256) across the full range and keeps any smaller-scale flow in range near -1.
    Out-of-range coordinates are exactly where sampler implementations legitimately disagree
    (border/zero handling), so an in-range grid is required for a cross-runtime-comparable repro."""
    sc = cn + "__gnsc"; of = cn + "__gnof"; mu = cn + "__gnmul"; nm = cn + "__gnorm"
    onnx_nodes += [
        helper.make_node("Constant", [], [sc], name=sc + "_c",
                         value=numpy_helper.from_array(np.array(1.0 / 128.0, np.float16), name=sc + "_v")),
        helper.make_node("Constant", [], [of], name=of + "_c",
                         value=numpy_helper.from_array(np.array(1.0, np.float16), name=of + "_v")),
        helper.make_node("Mul", [cn, sc], [mu], name=mu + "_m"),
        helper.make_node("Sub", [mu, of], [nm], name=nm + "_s"),
    ]
    interp.shp[mu] = tgt
    interp.shp[nm] = tgt
    return nm


def _rewire_gridsample(interp, nn, new_in, onnx_nodes, flow_src):
    """Rebuild a GridSample grid to the shape its data demands, while KEEPING the conv-derived flow.

    Each warp block builds its sampling grid from a coordinate grid (baked `arange` axes + `[1,H,W,1]`
    reshape/expand constants) added to the optical flow (the conv/ConvTranspose output). Those baked
    grid constants do NOT survive the SSA-collapsed dump, and their spatial (arange length) rarely
    matches the flow's, so the original grid subgraph is shape-inconsistent. GridSample's contract
    fixes the grid shape though: for NCHW data `[N,C,Hd,Wd]` the grid is `[N,Hd,Wd,2]`. We synthesise
    exactly that from the block's FLOW tensor (a 2-channel conv output) via Resize->Transpose, so the
    grid stays a function of the convolutions (the outputs still depend on the whole network) and is
    shape-correct. Only the grid's exact coordinate values change -- the op/dtype/shape structure and
    the conv dependency are preserved, which is the repro contract for random-weight rebuilds.
    """
    dsh = interp.gs(new_in[0]) if new_in and new_in[0] else None
    if not concrete(dsh) or len(dsh) != 4:
        return new_in
    n, _, hd, wd = dsh
    base = nn["out"][0]
    tgt = [n, hd, wd, 2]
    # The grid must be [N,Hd,Wd,2]. Build it from a CONV feature so the outputs keep depending on the
    # network, using ONLY Slice / Transpose / Reshape / Cast (no Resize -- a Resize here would change
    # the channel count and force a CPU fallback). Prefer the conv-derived optical-flow tensor whose
    # spatial already matches the GridSample data (the flow is [N,2,H,W] or [N,H,W,2] at Hd x Wd); else
    # fall back to a zeros grid (shape-correct, conv-independent). Only the grid VALUES change.
    fsh = interp.gs(flow_src) if flow_src is not None else None
    if flow_src is not None and concrete(fsh) and len(fsh) == 4:
        cn = base + "__gcast"
        if fsh[3] == 2 and fsh[1] == hd and fsh[2] == wd:
            # already [N,Hd,Wd,2] -> just cast to fp16
            onnx_nodes.append(helper.make_node("Cast", [flow_src], [cn], name=cn + "_c",
                                               to=int(TensorProto.FLOAT16)))
            interp.shp[cn] = tgt
            return [new_in[0], _norm_grid(interp, onnx_nodes, base, cn, tgt)]
        if fsh[1] >= 2 and fsh[2] == hd and fsh[3] == wd:
            # [N,C,Hd,Wd] NCHW -> Slice first 2 channels, Transpose to [N,Hd,Wd,2], Cast
            sl = base + "__gsl"; st = base + "__gst"; en = base + "__gen"; ax = base + "__gax"
            onnx_nodes += [
                helper.make_node("Constant", [], [st], name=st + "_c",
                                 value=numpy_helper.from_array(np.array([0], np.int64), name=st + "_v")),
                helper.make_node("Constant", [], [en], name=en + "_c",
                                 value=numpy_helper.from_array(np.array([2], np.int64), name=en + "_v")),
                helper.make_node("Constant", [], [ax], name=ax + "_c",
                                 value=numpy_helper.from_array(np.array([1], np.int64), name=ax + "_v")),
                helper.make_node("Slice", [flow_src, st, en, ax], [sl], name=sl + "_s"),
            ]
            tn = base + "__gtrans"
            onnx_nodes.append(helper.make_node("Transpose", [sl], [tn], name=tn + "_t", perm=[0, 2, 3, 1]))
            onnx_nodes.append(helper.make_node("Cast", [tn], [cn], name=cn + "_c", to=int(TensorProto.FLOAT16)))
            interp.shp[sl] = [n, 2, hd, wd]
            interp.shp[tn] = tgt
            interp.shp[cn] = tgt
            return [new_in[0], _norm_grid(interp, onnx_nodes, base, cn, tgt)]
        if fsh[1] >= 2 and len(fsh) == 4:
            # NCHW flow at a DIFFERENT spatial -> Slice 2 channels, SPATIAL Resize to Hd x Wd (channels
            # unchanged, so this counts as a genuine spatial resize like the model's own), then Transpose.
            sl = base + "__gsl"; st = base + "__gst"; en = base + "__gen"; ax = base + "__gax"
            onnx_nodes += [
                helper.make_node("Constant", [], [st], name=st + "_c",
                                 value=numpy_helper.from_array(np.array([0], np.int64), name=st + "_v")),
                helper.make_node("Constant", [], [en], name=en + "_c",
                                 value=numpy_helper.from_array(np.array([2], np.int64), name=en + "_v")),
                helper.make_node("Constant", [], [ax], name=ax + "_c",
                                 value=numpy_helper.from_array(np.array([1], np.int64), name=ax + "_v")),
                helper.make_node("Slice", [flow_src, st, en, ax], [sl], name=sl + "_s"),
            ]
            rz = base + "__gsp"; szc = base + "__gspsz"
            onnx_nodes.append(helper.make_node(
                "Constant", [], [szc], name=szc + "_c",
                value=numpy_helper.from_array(np.array([n, 2, hd, wd], dtype=np.int64), name=szc + "_v")))
            onnx_nodes.append(helper.make_node("Resize", [sl, "", "", szc], [rz], name=rz + "_r", mode="nearest"))
            tn = base + "__gtrans"
            onnx_nodes.append(helper.make_node("Transpose", [rz], [tn], name=tn + "_t", perm=[0, 2, 3, 1]))
            onnx_nodes.append(helper.make_node("Cast", [tn], [cn], name=cn + "_c", to=int(TensorProto.FLOAT16)))
            interp.shp[sl] = [n, 2, fsh[2], fsh[3]]
            interp.shp[rz] = [n, 2, hd, wd]
            interp.shp[tn] = tgt
            interp.shp[cn] = tgt
            return [new_in[0], _norm_grid(interp, onnx_nodes, base, cn, tgt)]
    # fallback: derive the grid from the GridSample DATA (a conv feature already at [N,C,Hd,Wd]) so the
    # outputs keep depending on the convolutions -- Slice 2 channels, Transpose to [N,Hd,Wd,2], Cast.
    # No Resize (spatial already matches). Zeros only if the data has <2 channels.
    dc = dsh[1]
    if dc >= 2:
        sl = base + "__gdsl"; st = base + "__gdst"; en = base + "__gden"; ax = base + "__gdax"
        onnx_nodes += [
            helper.make_node("Constant", [], [st], name=st + "_c",
                             value=numpy_helper.from_array(np.array([0], np.int64), name=st + "_v")),
            helper.make_node("Constant", [], [en], name=en + "_c",
                             value=numpy_helper.from_array(np.array([2], np.int64), name=en + "_v")),
            helper.make_node("Constant", [], [ax], name=ax + "_c",
                             value=numpy_helper.from_array(np.array([1], np.int64), name=ax + "_v")),
            helper.make_node("Slice", [new_in[0], st, en, ax], [sl], name=sl + "_s"),
        ]
        tn = base + "__gdtr"; cn = base + "__gdcast"
        onnx_nodes.append(helper.make_node("Transpose", [sl], [tn], name=tn + "_t", perm=[0, 2, 3, 1]))
        onnx_nodes.append(helper.make_node("Cast", [tn], [cn], name=cn + "_c", to=int(TensorProto.FLOAT16)))
        interp.shp[sl] = [n, 2, hd, wd]
        interp.shp[tn] = tgt
        interp.shp[cn] = tgt
        return [new_in[0], _norm_grid(interp, onnx_nodes, base, cn, tgt)]
    grid = np.zeros((n, hd, wd, 2), dtype=np.float16)
    gname = base + "__grid"
    onnx_nodes.append(helper.make_node("Constant", [], [gname], name=gname + "_c",
                                       value=numpy_helper.from_array(grid, name=gname + "_v")))
    interp.val[gname] = grid
    interp.shp[gname] = tgt
    return [new_in[0], gname]


def _resize_targets(nodes):
    """Intended (Ht, Wt) per Resize node index, when recoverable.

    Default (absent here) is a factor-2 downscale. Overrides encode the two known cases:
      * the full-resolution input downscales go to the half-resolution working res;
      * flow-decoder Resizes with a FLOAT `scales` operand are 2x UPSAMPLES.
    """
    tgt = {}
    for i, nd in enumerate(nodes):
        if nd["op_type"] != "Resize":
            continue
        uses_scales = (len(nd["inputs"]) > 2 and nd["inputs"][2]) and not (
            len(nd["inputs"]) > 3 and nd["inputs"][3])
        if uses_scales:
            tgt[i] = ("up2", "up2")  # sentinel -> upsample x2 (resolved against X in _rewire_resize)
    return tgt


def _rewire_resize(interp, nn, new_in, onnx_nodes, resize_target):
    """Give a Resize a concrete, self-consistent output size.

    The dumped size vectors are `Concat(Slice(Shape(X)[:2]), baked[H,W])` (sizes idiom) or a FLOAT
    `scales` -- neither survives the SSA collapse. We recompute X's shape and rewire the Resize to a
    fresh INT64 `sizes` initializer `[N, C, Ht, Wt]`. Ht,Wt come from `resize_target` when the model's
    intended spatial for this node is known (keyed by original node index), else a factor-2 downscale
    (the pyramid default). Keeping N,C from X preserves the channel geometry that Conv/Concat need.
    """
    xsh = interp.gs(new_in[0]) if new_in and new_in[0] else None
    if not concrete(xsh) or len(xsh) != 4:
        return new_in  # can't recompute -> leave as dumped (may still work via computed sizes)
    idx = int(nn["name"].rsplit("__n", 1)[1])
    tgt = resize_target.get(idx)
    if tgt == ("up2", "up2"):
        ht, wt = xsh[2] * 2, xsh[3] * 2                       # flow-decoder upsample
    elif tgt is not None:
        ht, wt = tgt
    else:
        ht, wt = max(1, xsh[2] // 2), max(1, xsh[3] // 2)     # pyramid downscale (default)
    sizes = np.array([xsh[0], xsh[1], ht, wt], dtype=np.int64)
    sname = nn["out"][0] + "__sizes"
    onnx_nodes.append(helper.make_node("Constant", [], [sname], name=sname + "_c",
                                       value=numpy_helper.from_array(sizes, name=sname + "_v")))
    interp.val[sname] = sizes
    interp.shp[sname] = [4]
    # rewire: keep X (0), drop roi/scales (1,2), set sizes (3)
    return [new_in[0], "", "", sname]


def _chain_to_gather(uo, rec_by_out, const_node, interp, max_hops=4):
    """Walk value-preserving hops (Cast/Unsqueeze/Squeeze/Identity) from `uo` to a
    Gather(known-shape-vector, our-index-const); returns (gather_record, vector) or (None, None)."""
    r = rec_by_out.get(uo)
    hops = 0
    while r is not None and r["op"] in ("Cast", "Unsqueeze", "Squeeze", "Identity") and hops < max_hops:
        uo = r["in"][0]
        r = rec_by_out.get(uo)
        hops += 1
    if r is not None and r["op"] == "Gather" and len(r["in"]) > 1 and r["in"][1] in const_node:
        vdata = interp.gv(r["in"][0])
        if vdata is not None:
            return r, np.atleast_1d(vdata).astype(np.int64)
    return None, None


def _impose_vec(uo, T, rec_by_out, const_node, set_const, interp, log, depth=0):
    """FAITHFUL mode: make INT64 tensor `uo` evaluate to T at runtime by patching only the leaf
    Constants we materialised. Inverts the shape-machinery idioms (Concat / Cast / Unsqueeze /
    Squeeze / Gather-from-shape / scalar Mul-Div-Add-Sub / leading-dims Slice); computed parts whose
    interpreted value already equals the wanted slice are left untouched. Returns True when the
    imposition is complete (every leaf reachable and patched/consistent)."""
    T = np.atleast_1d(np.asarray(T)).astype(np.int64)
    if depth > 12:
        log(f"impose: depth cap at {uo}")
        return False
    if uo in const_node:
        dtn, shp = const_node[uo][1], const_node[uo][2]
        set_const(uo, T.reshape(shp) if shp else np.asarray(T[0]), dtn)
        return True
    r = rec_by_out.get(uo)
    if r is None:
        cur = interp.gv(uo)
        ok = cur is not None and np.array_equal(np.atleast_1d(cur).astype(np.int64), T)
        if not ok:
            log(f"impose: {uo} unpatchable (input/init), want {T.tolist()}")
        return ok
    op, ins = r["op"], r["in"]
    if op in ("Cast", "Unsqueeze", "Squeeze", "Identity"):
        return _impose_vec(ins[0], T, rec_by_out, const_node, set_const, interp, log, depth + 1)
    if op == "Concat":
        parts = [p for p in ins if p]
        lens = []
        for p in parts:
            v = interp.gv(p)
            if v is not None:
                lens.append(np.atleast_1d(v).size)
            else:
                s = interp.gs(p)
                lens.append(s[0] if s is not None and len(s) == 1 else None)
        if None in lens or sum(lens) != T.size:
            # a Slice part with patchable bounds can absorb the residual length (the
            # Concat(Slice(Shape(X))[:k], baked) sizes idiom before its bounds are solved)
            def adjustable(p):
                pr = rec_by_out.get(p)
                return pr is not None and pr["op"] == "Slice" and len(pr["in"]) > 2 and pr["in"][2] in const_node
            adj = [k for k, p in enumerate(parts) if adjustable(p)]
            fixed = sum(L for k, L in enumerate(lens) if k not in adj and L is not None)
            if len(adj) == 1 and all(lens[k] is not None for k in range(len(parts)) if k not in adj) \
                    and T.size - fixed >= 1:
                lens[adj[0]] = T.size - fixed
            else:
                log(f"impose: Concat {uo} part lengths {lens} vs target {T.tolist()}")
                return False
        ok, off = True, 0
        for p, L in zip(parts, lens):
            want = T[off:off + L]
            off += L
            v = interp.gv(p)
            if p in const_node:
                dtn, shp = const_node[p][1], const_node[p][2]
                set_const(p, want.reshape(shp) if shp else np.asarray(want[0]), dtn)
            elif v is not None and np.array_equal(np.atleast_1d(v).astype(np.int64), want):
                continue
            else:
                ok = _impose_vec(p, want, rec_by_out, const_node, set_const, interp, log, depth + 1) and ok
        return ok
    if op == "Gather":
        # scalar/vector picked from a known shape vector: choose indices where data == want
        vdata = interp.gv(ins[0])
        if vdata is not None and len(ins) > 1 and ins[1] in const_node:
            flat = np.atleast_1d(vdata).astype(np.int64)
            idx = []
            for w in T:
                hit = np.nonzero(flat == w)[0]
                if hit.size == 0:
                    log(f"impose: Gather {uo} wants {w} not present in {flat.tolist()}")
                    return False
                idx.append(int(hit[0]))
            dtn, shp = const_node[ins[1]][1], const_node[ins[1]][2]
            arr = np.asarray(idx[0]) if not shp else np.asarray(idx, dtype=np.int64).reshape(shp)
            set_const(ins[1], arr, dtn)
            return True
        return _impose_vec(ins[0], T, rec_by_out, const_node, set_const, interp, log, depth + 1) if T.size > 1 else False
    if op in ("Mul", "Div", "Add", "Sub"):
        a, b = ins[0], ins[1]
        va, vb = interp.gv(a), interp.gv(b)
        # composite Gather-from-shape * scalar-const solve FIRST: Div(Gather(Shape(X)), c) = T has
        # two unknowns (dim index, divisor); pick the dim s and smallest integer c with s/c == T
        # (resp. s*c == T for Mul). The single-operand inversions below would otherwise lock the
        # placeholder Gather value in and derive a poisoned (even zero) scalar.
        if T.size == 1 and op in ("Mul", "Div"):
            t = int(T[0])
            for x, other in ((a, b), (b, a)):
                if other not in const_node:
                    continue
                g, vec = _chain_to_gather(x, rec_by_out, const_node, interp)
                if g is None:
                    continue
                best = None
                for c in range(1, 65):
                    if op == "Div":
                        s = t * c
                    elif t % c == 0:
                        s = t // c
                    else:
                        continue
                    hit = np.nonzero(vec == s)[0]
                    if hit.size:
                        best = (int(hit[0]), c)
                        break
                if best is None:
                    continue
                idx, c = best
                idt, ishp = const_node[g["in"][1]][1], const_node[g["in"][1]][2]
                set_const(g["in"][1], np.asarray(idx) if not ishp
                          else np.full(ishp, idx, dtype=np.int64), idt)
                dtn, shp = const_node[other][1], const_node[other][2]
                cv = float(c) if dtn in ("FLOAT", "FLOAT16", "DOUBLE") else c
                set_const(other, np.asarray(cv) if not shp else np.full(shp, cv), dtn)
                return True
        for x, other_v, left in ((a, vb, True), (b, va, False)):
            if other_v is None:
                continue
            o = np.atleast_1d(other_v).astype(np.float64)
            Tf = T.astype(np.float64)
            if op == "Mul":
                val = Tf / o
            elif op == "Add":
                val = Tf - o
            elif op == "Div":            # result = A / B
                val = Tf * o if left else o / Tf
            else:                        # Sub: result = A - B
                val = Tf + o if left else o - Tf
            tgt = x
            if tgt in const_node:
                dtn, shp = const_node[tgt][1], const_node[tgt][2]
                out = val if dtn in ("FLOAT", "FLOAT16", "DOUBLE") else np.round(val)
                if dtn not in ("FLOAT", "FLOAT16", "DOUBLE") and np.any(out <= 0):
                    continue  # a zero/negative INT64 shape term poisons every consumer downstream
                set_const(tgt, out.reshape(shp) if shp else np.asarray(out.ravel()[0]), dtn)
                return True
            if _impose_vec(tgt, np.round(val).astype(np.int64), rec_by_out, const_node, set_const,
                           interp, log, depth + 1):
                return True
        log(f"impose: {op} {uo} has no invertible operand")
        return False
    if op == "Slice":
        # leading-dims window: patch OUR bounds to [0:len(T)] on axis 0, then require the data's
        # leading dims to already equal T (they come from Shape(X); X's shape is not patchable).
        for pos, val in ((1, np.zeros(1, np.int64)), (2, np.full(1, T.size, np.int64)),
                         (3, np.zeros(1, np.int64)), (4, np.ones(1, np.int64))):
            if len(ins) > pos and ins[pos] in const_node:
                dtn, shp = const_node[ins[pos]][1], const_node[ins[pos]][2]
                set_const(ins[pos], val.reshape(shp) if shp else np.asarray(val[0]), dtn)
        vdata = interp.gv(ins[0])
        if vdata is not None and np.array_equal(np.atleast_1d(vdata).astype(np.int64)[:T.size], T):
            return True
        log(f"impose: Slice {uo} leading dims {None if vdata is None else np.atleast_1d(vdata).tolist()} != {T.tolist()}")
        return False
    log(f"impose: cannot invert {op} at {uo}")
    return False


_ELEMWISE = {"Mul", "Add", "Sub", "Div", "Where", "Greater", "GreaterOrEqual", "Equal", "Min",
             "Max", "Pow", "Cast", "Clip", "Reciprocal", "Neg", "Sigmoid", "Sqrt", "Relu",
             "PRelu", "Identity"}


def _bcast_ok(shapes):
    """True if the concrete shapes are mutually NumPy-broadcastable."""
    n = max(len(s) for s in shapes)
    for k in range(1, n + 1):
        dims = {s[len(s) - k] for s in shapes if len(s) >= k and s[len(s) - k] != 1}
        if len(dims) > 1:
            return False
    return True


def _repair_misbindings(new_nodes, onnx_by_name, producers, rec_by_out, interp, log,
                        max_fixes=40, max_depth=8):
    """FAITHFUL mode: the trace reuses tensor names, and nearest-preceding binding can wire a
    consumer to the WRONG producer (e.g. a later warp's GridSample output hijacks an earlier warp's
    collapsed name). Such mistakes surface as provable broadcast clashes at elementwise ops. For each clash,
    walk up the operand chains; at every edge whose ORIGINAL name has alternative producers, try
    the other candidates (nearest first) and keep the rewiring that makes the clash broadcastable
    (verified by a full interpreter replay). Structure is untouched -- only which producer an edge
    references changes."""
    def uidx(u):
        r = rec_by_out.get(u)
        return r["idx"] if r is not None else -1

    def find_clash(skip):
        for r in new_nodes:
            if id(r) in skip:
                continue
            if r["op"] not in ("Mul", "Add", "Sub", "Div", "Where", "Greater", "GreaterOrEqual",
                               "Equal", "Min", "Max", "Pow"):
                continue
            shapes = [interp.gs(u) for u in r["in"] if u]
            if len(shapes) < 2 or any(not concrete(s) for s in shapes):
                continue
            if not _bcast_ok(shapes):
                return r
        return None

    def dump_chain(uo, depth=0, lines=None):
        lines = [] if lines is None else lines
        r = rec_by_out.get(uo)
        if r is None:
            lines.append("  " * depth + f"{uo} (leaf) {interp.gs(uo)}")
            return lines
        lines.append("  " * depth + f"{uo} <- {r['op']} {interp.gs(uo)}")
        if depth < 4:
            for u in r["in"]:
                if u:
                    dump_chain(u, depth + 1, lines)
        return lines

    def rewire(node, pos, u):
        node["in"][pos] = u
        onnx_by_name[node["name"]].input[pos] = u

    def try_fix(clash):
        seen, queue = set(), [(clash, 0)]
        while queue:
            node, depth = queue.pop(0)
            if id(node) in seen or depth > max_depth:
                continue
            seen.add(id(node))
            for pos, u in enumerate(node["in"]):
                if not u:
                    continue
                orig = node["orig"]["inputs"][pos] if pos < len(node["orig"]["inputs"]) else None
                cands = [c for c in (producers.get(orig) or []) if c != u and uidx(c) < node["idx"]]
                for c in reversed(cands[-4:])  :
                    old = node["in"][pos]
                    rewire(node, pos, c)
                    _replay_interp(interp, new_nodes)
                    shapes = [interp.gs(x) for x in clash["in"] if x]
                    if all(concrete(s) for s in shapes) and _bcast_ok(shapes):
                        log(f"rebind: {node['name']} operand {pos} {old} -> {c}")
                        return True
                    rewire(node, pos, old)
                pr = rec_by_out.get(u)
                if pr is not None and pr["op"] in _ELEMWISE:
                    queue.append((pr, depth + 1))
        _replay_interp(interp, new_nodes)
        return False

    skip = set()
    for _ in range(max_fixes):
        _replay_interp(interp, new_nodes)
        clash = find_clash(skip)
        if clash is None:
            return
        if not try_fix(clash):
            log(f"rebind: unresolvable clash at {clash['name']} "
                f"{[interp.gs(u) for u in clash['in'] if u]}")
            for u in clash["in"]:
                if u:
                    for ln in dump_chain(u):
                        log("rebind:   " + ln)
            skip.add(id(clash))


def _fix_interior_reshape_counts(new_nodes, rec_by_out, const_node, set_const, interp, log):
    """FAITHFUL mode, last stage: an interior Reshape whose machinery-computed target is
    count-inconsistent with its (now settled) data gets its free all-1 constant Concat slots scaled
    to absorb the residual ratio (a space-to-depth factor whose literal dims the dump lost)."""
    for r in new_nodes:
        if r["op"] != "Reshape" or len(r["in"]) < 2 or not r["in"][1]:
            continue
        dsh = interp.gs(r["in"][0])
        tgt = interp.gv(r["in"][1])
        if not concrete(dsh) or tgt is None:
            continue
        tgt = np.atleast_1d(tgt).astype(np.int64)
        if np.any(tgt <= 0) or _reshape_valid(list(tgt), dsh):
            continue
        total, have = int(np.prod(dsh)), int(np.prod(tgt))
        if have == 0 or total % have:
            continue
        residual = total // have
        cr = rec_by_out.get(r["in"][1])
        if cr is None or cr["op"] != "Concat":
            continue
        free = [p for p in cr["in"] if p and p in const_node
                and interp.gv(p) is not None and np.all(np.atleast_1d(interp.gv(p)) == 1)]
        if not free:
            continue
        # distribute the residual over the free slots: even power split when possible, else all
        # into the first slot
        vals = [1] * len(free)
        k = len(free)
        root = round(residual ** (1.0 / k))
        if root ** k == residual:
            vals = [root] * k
        else:
            vals[0] = residual
        for p, v in zip(free, vals):
            dtn, shp = const_node[p][1], const_node[p][2]
            set_const(p, np.full(shp, v, dtype=np.int64) if shp else np.asarray(v), dtn)
        log(f"count-fix: Reshape {r['name']} residual {residual} -> slots {vals}")
        _replay_interp(interp, new_nodes)


def _faithful_ort_repair(model, log, max_iter=60):
    """FAITHFUL mode: make the structure-preserving graph loadable by ORT without touching any node
    or edge. The dump loses interior resolution-ladder values, so a synthesized constant can clash
    with a real anchor (e.g. an index-ramp initializer) in ORT's static checker. On each
    ShapeInferenceError, neutralise ONE float Constant/initializer operand of the offending node to
    a scalar (dtype kept, value = current mean). INT64 shape-control operands are never touched."""
    import onnxruntime as ort
    for _ in range(max_iter):
        try:
            ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            return model
        except Exception as e:
            msg = str(e)
            m = re.search(r"Node \(([^)]+)\)", msg)
            if not m:
                log(f"ort-repair: unparseable failure: {msg[:200]}")
                return model
            node = next((n for n in model.graph.node if n.name == m.group(1)), None)
            if node is None:
                log(f"ort-repair: node {m.group(1)} not found")
                return model
            fixed = False
            inits = {i.name: i for i in model.graph.initializer}
            consts = {n.output[0]: n for n in model.graph.node
                      if n.op_type == "Constant" and n.output}
            for opname in node.input:
                if opname in inits:
                    t = inits[opname]
                    if t.data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE):
                        continue
                    arr = numpy_helper.to_array(t)
                    scal = np.asarray(arr.ravel().mean() if arr.size else 0, dtype=arr.dtype)
                    inits[opname].CopyFrom(numpy_helper.from_array(scal.reshape(()), name=opname))
                    log(f"ort-repair: initializer {opname} {list(arr.shape)} -> scalar (node {node.name})")
                    fixed = True
                    break
                if opname in consts:
                    cn = consts[opname]
                    val = next((a for a in cn.attribute if a.name == "value"), None)
                    if val is None or val.t.data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16,
                                                              TensorProto.DOUBLE):
                        continue
                    arr = numpy_helper.to_array(val.t)
                    scal = np.asarray(arr.ravel().mean() if arr.size else 0, dtype=arr.dtype)
                    del cn.attribute[:]
                    cn.attribute.append(helper.make_attribute(
                        "value", numpy_helper.from_array(scal.reshape(()), name=opname + "_v")))
                    log(f"ort-repair: constant {opname} {list(arr.shape)} -> scalar (node {node.name})")
                    fixed = True
                    break
            if not fixed:
                log(f"ort-repair: no neutralisable operand on {node.name} ({node.op_type}): {msg[:160]}")
                return model
    return model


def _replay_interp(interp, replay_records):
    """Recompute interpreter values over the emitted records in order (Constant values persist in
    interp.val, so patched leaves propagate through the machinery)."""
    for r in replay_records:
        if r["op"] == "Constant":
            continue
        interp.run(r["op"], r["in"], r["out"], r["attrs"])


def _faithful_resize(interp, nn, new_in, rec_by_out, const_node, set_const, rec_out, log):
    """FAITHFUL mode: keep the Resize's dumped operands; back-solve the constants in its sizes /
    scales chain so the runtime machinery computes a valid, baked-mode-equivalent geometry
    (recorded output spatial when concrete, else the pyramid factor-2 downscale; float-scales
    Resizes are 2x upsamples, matching _resize_targets)."""
    xsh = interp.gs(new_in[0]) if new_in and new_in[0] else None
    has_sizes = len(new_in) > 3 and new_in[3]
    has_scales = len(new_in) > 2 and new_in[2]
    if has_scales and not has_sizes:
        if new_in[2] not in const_node:
            log(f"impose: Resize {nn['name']} computed scales left as-is")
            return "computed-scales"
        if concrete(xsh) and len(xsh) == 4 and concrete(rec_out) and len(rec_out) == 4 and rec_out[2] > 0:
            fh, fw = rec_out[2] / xsh[2], rec_out[3] / xsh[3]
            mode = "recout-scales"
        else:
            fh = fw = 2.0  # flow-decoder upsample (see _resize_targets)
            mode = "default-scales"
        dtn, shp = const_node[new_in[2]][1], const_node[new_in[2]][2]
        set_const(new_in[2], np.array([1.0, 1.0, fh, fw]).reshape(shp if shp else (4,)), dtn)
        return mode
    if not concrete(xsh) or len(xsh) != 4:
        log(f"impose: Resize {nn['name']} data shape unknown; leaving chain as-is")
        return "skipped"
    if has_sizes:
        if concrete(rec_out) and len(rec_out) == 4:
            ht, wt = int(rec_out[2]), int(rec_out[3])
            mode = "recout-sizes"
        else:
            ht, wt = max(1, xsh[2] // 2), max(1, xsh[3] // 2)
            mode = "default-sizes"
        T = [xsh[0], xsh[1], ht, wt]
        _impose_vec(new_in[3], T, rec_by_out, const_node, set_const, interp, log)
        return mode
    return "no-operand"


def _fix_shape_control(interp, op, ins, attrs, const_node, set_const, rec_out=None, faithful=False):
    """Right before a shape-transforming op runs, patch its Constant shape-control inputs so the
    geometry is self-consistent. Preference order for a Reshape/Expand target:
      1. the node's RECORDED output shape (`rec_out`) when it is concrete and element-count-consistent
         -- this is the true target the exporter baked (reliable for uniquely-named outputs);
      2. a geometry heuristic derived from the (now-known) data input shape.
    Only constants we materialised are patched; a computed (Shape/Slice/Concat) size vector is trusted.
    """
    def is_ours(pos):
        return pos < len(ins) and ins[pos] in const_node

    def dt(pos):
        return const_node[ins[pos]][1]

    def cur(pos):
        return interp.gv(ins[pos])

    if op == "Reshape":
        dsh = interp.gs(ins[0])
        if is_ours(1) and concrete(dsh):
            tgt_len = len(np.atleast_1d(cur(1)))
            total = int(np.prod(dsh)) if dsh else 1
            drank = len(dsh)
            new = None
            # 1) recorded output shape, if concrete + same element count + same rank as the target
            if concrete(rec_out) and len(rec_out) == tgt_len and int(np.prod(rec_out)) == total:
                new = np.array(rec_out, dtype=np.int64)
            if new is None:
                if tgt_len == drank:
                    new = np.array(dsh, dtype=np.int64)                    # identity reshape
                elif drank <= 1 and tgt_len >= 2:
                    # coordinate/normalisation vector (len L) -> broadcast on CHANNEL axis: [1,L,1,...]
                    new = np.array([1, total] + [1] * (tgt_len - 2), dtype=np.int64)
                elif tgt_len > drank:
                    # unsqueeze-like: prepend leading 1s, keep the data's dims (NCHW-preserving).
                    new = np.array([1] * (tgt_len - drank) + list(dsh), dtype=np.int64)
                else:
                    # squeeze-like: fold leading dims into dim 0, keep the trailing (tgt_len-1) dims.
                    keep = list(dsh)[drank - (tgt_len - 1):] if tgt_len > 1 else []
                    lead = total // int(np.prod(keep)) if keep else total
                    new = np.array([lead] + keep, dtype=np.int64)
            set_const(ins[1], new, dt(1))
        return

    if op == "Expand":
        dsh = interp.gs(ins[0])
        if is_ours(1) and concrete(dsh):
            tgt_len = len(np.atleast_1d(cur(1)))
            base = list(dsh)
            if concrete(rec_out) and len(rec_out) == tgt_len and _broadcastable(base, rec_out):
                new = list(rec_out)                                        # recorded expand target
            elif tgt_len >= len(base):
                new = [1] * (tgt_len - len(base)) + base                   # left-pad with 1s (identity)
            else:
                new = base[len(base) - tgt_len:]
            set_const(ins[1], np.array(new, dtype=np.int64), dt(1))
        return

    if op == "Resize":
        # handled specially by the caller (needs to rewire the sizes input) -- see _rewire_resize.
        return

    if op == "Slice":
        dsh = interp.gs(ins[0])
        if not concrete(dsh):
            return
        rank = len(dsh)
        # recorded output dims are a real anchor for OUR placeholder bounds: where a concrete
        # rec_out dim shrinks the input dim, window that axis to [0:rec_out[i]]. (The full-range
        # placeholder otherwise silently keeps the dim and poisons downstream broadcasts.)
        if faithful and rec_out is not None and len(rec_out) == rank and is_ours(1) and is_ours(2):
            axes = [i for i in range(rank)
                    if isinstance(rec_out[i], int) and rec_out[i] > 0 and rec_out[i] != dsh[i]]
            k1 = len(np.atleast_1d(cur(1)))
            if axes and len(axes) <= k1:
                pad = k1 - len(axes)                        # spare bound slots -> no-op windows
                spare = [i for i in range(rank) if i not in axes][:pad]
                axv = axes + spare
                set_const(ins[1], np.zeros(k1, np.int64), dt(1))
                set_const(ins[2], np.array([rec_out[i] if i in axes else dsh[i] for i in axv],
                                           dtype=np.int64), dt(2))
                if len(ins) > 3 and is_ours(3):
                    set_const(ins[3], np.array(axv, dtype=np.int64), dt(3))
                if len(ins) > 4 and is_ours(4):
                    set_const(ins[4], np.ones(k1, np.int64), dt(4))
                return
        # axes (index 3): clamp to valid range so ends/starts apply to real dims
        if len(ins) > 3 and is_ours(3):
            k = len(np.atleast_1d(cur(3)))
            set_const(ins[3], np.arange(min(k, rank), dtype=np.int64)[:k] if k <= rank
                      else np.arange(rank, dtype=np.int64), dt(3))
        # starts -> 0, ends -> +inf, steps -> 1 (already set that way; re-affirm length safety)
        return

    if op in ("Conv", "ConvTranspose", "GridSample", "Concat", "Gather", "Unsqueeze", "Squeeze",
              "ConstantOfShape", "ReduceSum", "Where", "Equal", "Transpose"):
        return
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("-o", "--out", default="rebuilt.onnx")
    ap.add_argument("--inputs-dir", default=None, help="write concrete input tensors here (.npy + .bin)")
    ap.add_argument("--goldens-dir", default=None, help="write ORT golden outputs here (.npy + .bin)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="run onnxruntime to validate + dump goldens")
    ap.add_argument("--faithful", action="store_true",
                    help="structure-preserving rebuild: keep every node and the runtime shape "
                         "machinery (Shape/Slice/Concat/... chains feeding Resize/Reshape/Expand); "
                         "instead of rewiring shape-consumer operands, back-solve the leaf Constant "
                         "VALUES so the machinery computes self-consistent geometry at runtime. "
                         "No dead-node pruning, no repair/coercion passes.")
    ap.add_argument("--raw-names", action="store_true",
                    help="with --faithful: emit nodes under their ORIGINAL dumped tensor names, "
                         "duplicates included (the trace reuses names, so the file violates SSA "
                         "exactly like the source export does). onnx.checker and ORT reject such "
                         "a file -- it exists to exercise a consumer's own SSA-rename/binding, "
                         "not to run in ORT. Constant values come from the interpreted pass.")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    d = json.load(open(args.json))
    nodes = d["nodes"]
    inits_meta = {i["name"]: i for i in d["initializers"]}
    input_names = {i["name"] for i in d["inputs"]}
    init_names = set(inits_meta)
    input_meta = {i["name"]: i for i in d["inputs"]}
    graph_out_names = [o["name"] for o in d["outputs"]]

    rec_shape = {t["name"]: t.get("shape") for t in d["tensors"]}

    def rec_rank(name):
        s = rec_shape.get(name)
        return len(s) if s is not None else None

    # ---- pre-pass: consumer roles per node output (naive nearest-preceding binding) ----
    # Used only to know how each Constant is consumed (op_type + input position) so it can be
    # synthesized correctly; the actual emission below uses rank-aware binding.
    _latest = {nm: nm for nm in input_names | init_names}
    role_consumers = {}  # uniq out name (naive) -> list of (naive_consumer_idx, op, pos)
    naive_out = []       # per node: its naive unique output names
    for i, nd in enumerate(nodes):
        outs = []
        for pos, s in enumerate(nd["inputs"]):
            if not s:
                continue
            u = _latest.get(s, s)
            role_consumers.setdefault(u, []).append((i, nd["op_type"], pos))
        for o in nd["outputs"]:
            uo = f"{o}__n{i}" if o else ""
            outs.append(uo)
            if o:
                _latest[o] = uo
        naive_out.append(outs)

    # ---- interpreter: seed input & weight shapes ----
    interp = Interp(rng)
    for nm, it in input_meta.items():
        interp.shp[nm] = list(it["shape"])
    for nm, meta in inits_meta.items():
        interp.shp[nm] = list(meta["shape"])

    # ---- build initializers (weights: FLOAT16) ----
    # Values are chosen NON-DEGENERATE so ORT produces varied, finite, non-zero outputs (a real
    # regression vehicle): convolution kernels get a small positive-mean Gaussian (positive mean keeps
    # post-ReLU/PReLU activations alive rather than collapsing to 0), and bias/1-D tensors get a small
    # positive offset (also anti-collapse). Magnitudes stay small to avoid fp16 overflow across depth.
    onnx_inits = []
    for nm, meta in inits_meta.items():
        npdt = NP_OF[meta["dtype"]]
        shp = meta["shape"]
        n = int(np.prod(shp)) if shp else 1
        rank = len(shp) if shp else 0
        if meta["dtype"] in ("FLOAT16", "FLOAT", "DOUBLE"):
            if rank <= 1:                                  # bias / PReLU-slope / 1-D param
                arr = (rng.standard_normal(n) * 0.01 + 0.02).astype(npdt)
            else:                                          # conv / linear weight
                arr = (rng.standard_normal(n) * 0.02 + 0.01).astype(npdt)
        else:
            arr = np.zeros(n, dtype=npdt)
        arr = arr.reshape(shp) if shp else arr.reshape(())
        onnx_inits.append(numpy_helper.from_array(arr, name=nm))
        interp.val[nm] = arr  # weights are known values (needed for conv weight-shape geometry)

    # ---- single forward pass: SSA-rename with RANK-AWARE binding + interpret + synth constants ----
    # The trace reuses names; a name's producers can have DIFFERENT ranks (e.g. "Mul_3_output_0" is
    # produced both as a rank-1 coordinate arange and as a rank-4 feature map). Nearest-preceding
    # binding then wires the wrong one. We disambiguate with the recorded collapsed rank of the name:
    # among preceding producers, prefer the nearest whose computed output rank matches that rank.
    producers = {}   # original name -> list of (uniq_id) in emission order
    consumers = {}   # uniq name -> list of (consumer_idx_in_new_nodes, op_type, input_pos)
    new_nodes = []
    onnx_nodes = []

    for nm in input_names | init_names:
        producers[nm] = [nm]

    def bind(s, consumer_op):
        """Nearest-preceding producer by default; rank-corrected only for feature maps.

        A reused name may have producers of different ranks -- e.g. "Mul_3_output_0" is emitted both
        as a rank-1 coordinate arange and as the rank-4 feature map. Nearest-preceding then wires the
        arange into a Conv/Resize/Shape. Only when the recorded collapsed rank is >=3 (unambiguously a
        feature map) and the nearest producer's rank disagrees do we walk back to the nearest producer
        whose rank matches. Scalar / shape (rank 0-1) names keep pure nearest-preceding, so the
        many-times-reused Constant_output_0 etc. bind to their true immediate producer.
        """
        cands = producers.get(s)
        if not cands:
            return s  # graph input / initializer / unseen -> own name
        nearest = cands[-1]
        want = rec_rank(s)
        if want is not None and want >= 3 and consumer_op in _FEATURE_CONSUMERS:
            r0 = interp.gs(nearest)
            # Override only when the nearest producer's rank is KNOWN and much smaller than the
            # feature rank (>=2 gap) -- i.e. a low-rank coordinate/arange has hijacked the name. An
            # unknown-rank nearest is left alone (more likely the true producer than a distant match);
            # a gap of 1 is a legit Gather/Reduce/Squeeze reduction.
            if r0 is not None and len(r0) <= want - 2:
                for u in reversed(cands):
                    r = interp.gs(u)
                    if r is not None and len(r) == want:
                        return u
        return nearest

    resize_target = _resize_targets(nodes)  # original node idx -> (Ht, Wt) when known
    const_node = {}  # uniq const name -> [onnx_node, dtype_name, recorded_valueattr_shape]
    rec_by_out = {}  # uniq out name -> emitted node record (faithful-mode back-solve)
    implog = []      # faithful-mode impose diagnostics
    faithful_resizes = []       # (nn, mode) -- default-guessed ones get consumer-anchored later
    faithful_out_reshapes = []  # (nn, declared shape) -- re-imposed after the ladder settles

    def ilog(msg):
        implog.append(msg)

    def set_const(uo, arr, dtn):
        """(Re)assign a materialised Constant's value in-place + refresh interp state."""
        arr = np.asarray(arr).astype(NP_OF[dtn])
        interp.val[uo] = arr
        interp.shp[uo] = list(arr.shape)
        nodeobj = const_node.get(uo)
        if nodeobj is not None:
            t = numpy_helper.from_array(arr, name=uo + "_v")
            del nodeobj[0].attribute[:]
            nodeobj[0].attribute.append(helper.make_attribute("value", t))

    for i, nd in enumerate(nodes):
        op = nd["op_type"]
        attrs = nd["attributes"]
        new_in = []
        for pos, s in enumerate(nd["inputs"]):
            if not s:
                new_in.append(""); continue
            u = bind(s, op)
            new_in.append(u)
            consumers.setdefault(u, []).append((len(new_nodes), op, pos))
        new_out = [(f"{o}__n{i}" if o else "") for o in nd["outputs"]]
        nn = {"op": op, "name": f"{nd['name']}__n{i}", "in": new_in, "out": new_out,
              "attrs": attrs, "orig": nd, "idx": len(new_nodes)}
        new_nodes.append(nn)
        for uo in new_out:
            if uo:
                rec_by_out[uo] = nn

        if op == "Constant":
            uo = new_out[0]
            dtn, shp = _const_meta(nd)
            if dtn is None:
                onnx_nodes.append(helper.make_node("Constant", [], [uo], name=nn["name"],
                                                   **_scalar_attrs(attrs)))
            else:
                # role from the naive consumer map (constants precede their consumers in the trace)
                cons = role_consumers.get(uo, [])
                roleset = {(cop, pos) for (_, cop, pos) in cons}
                val = interp.make_const(uo, dtn, shp, roleset)
                val = (val.reshape(shp) if shp else np.asarray(val).reshape(())).astype(NP_OF[dtn])
                interp.val[uo] = val
                interp.shp[uo] = list(val.shape)
                t = numpy_helper.from_array(val, name=uo + "_v")
                cnode = helper.make_node("Constant", [], [uo], name=nn["name"], value=t)
                onnx_nodes.append(cnode)
                const_node[uo] = [cnode, dtn, shp]
        elif op == "Resize":
            if args.faithful:
                ro = rec_shape.get(nd["outputs"][0]) if nd["outputs"] and nd["outputs"][0] else None
                fmode = _faithful_resize(interp, nn, new_in, rec_by_out, const_node, set_const,
                                         ro if concrete(ro) else None, ilog)
                faithful_resizes.append((nn, fmode))
                _replay_interp(interp, new_nodes)
            else:
                new_in = _rewire_resize(interp, nn, new_in, onnx_nodes, resize_target)
            interp.run(op, new_in, new_out, attrs)
            onnx_nodes.append(helper.make_node(op, new_in, new_out, name=nn["name"],
                                               **_clean_attrs(op, attrs)))
        elif op == "GridSample" and not args.faithful:
            by_out_now = {o: n for n in onnx_nodes for o in n.output if o}
            flow = _find_grid_flow(interp, new_in[1], by_out_now) if len(new_in) > 1 else None
            new_in = _rewire_gridsample(interp, nn, new_in, onnx_nodes, flow)
            interp.run(op, new_in, new_out, attrs)
            onnx_nodes.append(helper.make_node(op, new_in, new_out, name=nn["name"],
                                               **_clean_attrs(op, attrs)))
        elif op == "ConstantOfShape":
            # the dump records the fill `value` as a placeholder string; rebuild it as a real scalar
            # tensor of the recorded dtype (all INT64 here) so the output isn't the FLOAT default.
            dtn, _ = _const_meta(nd)
            interp.run(op, new_in, new_out, attrs)
            nd_attrs = {}
            if dtn:
                fill = numpy_helper.from_array(np.zeros((1,), dtype=NP_OF[dtn]))
                nd_attrs["value"] = fill
                # ConstantOfShape output dtype = fill dtype; refresh interp value/dtype
                ov = interp.gv(new_out[0])
                if ov is not None:
                    interp.val[new_out[0]] = ov.astype(NP_OF[dtn])
            onnx_nodes.append(helper.make_node(op, new_in, new_out, name=nn["name"], **nd_attrs))
        else:
            rec_out = rec_shape.get(nd["outputs"][0]) if nd["outputs"] and nd["outputs"][0] else None
            if op == "Reshape" and args.faithful:
                # keep the dumped wiring; back-solve constants so the runtime target is consistent.
                dsh0 = interp.gs(new_in[0])
                is_graph_out = bool(nd["outputs"]) and nd["outputs"][0] in set(graph_out_names)
                counts_ok = concrete(rec_out) and (
                    (not concrete(dsh0)) or int(np.prod(rec_out)) == int(np.prod(dsh0)))
                if len(new_in) > 1 and new_in[1]:
                    if new_in[1] in const_node:
                        _fix_shape_control(interp, op, new_in, attrs, const_node, set_const, rec_out, args.faithful)
                    else:
                        cur = interp.gv(new_in[1])
                        cur_valid = (cur is not None and concrete(dsh0) and
                                     _reshape_valid(list(np.atleast_1d(cur)), dsh0))
                        # graph outputs must land on the declared contract; interior targets are
                        # trusted when the machinery already computes a valid reshape.
                        if counts_ok and (is_graph_out or not cur_valid):
                            _impose_vec(new_in[1], rec_out, rec_by_out, const_node, set_const,
                                        interp, ilog)
                            _replay_interp(interp, new_nodes)
                        elif not cur_valid:
                            ilog(f"impose: Reshape {nn['name']} target unverifiable (rec_out "
                                 f"{rec_out}, data {dsh0})")
                        if is_graph_out:
                            faithful_out_reshapes.append((nn, rec_out))
            elif op == "Reshape":
                # Reshape targets in this graph are often COMPUTED (Shape/Gather/Concat) vectors that
                # collapse to wrong values, or materialised constants we synthesised as placeholders.
                # Recompute a valid target from: (1) the recorded output shape when concrete + count-
                # consistent; (2) for a declared graph output, its declared shape (coercing the data by
                # Resize if the count drifted); (3) otherwise a geometry heuristic over the known data
                # shape. We then pin the target via a fresh constant so the reshape always runs.
                dsh0 = interp.gs(new_in[0])
                tgt_len_c = None
                cur_tgt = interp.gv(new_in[1]) if len(new_in) > 1 and new_in[1] else None
                if cur_tgt is not None:
                    tgt_len_c = len(np.atleast_1d(cur_tgt))
                is_graph_out = bool(nd["outputs"]) and nd["outputs"][0] in set(graph_out_names)
                counts_ok = concrete(rec_out) and (
                    (not concrete(dsh0)) or int(np.prod(rec_out)) == int(np.prod(dsh0)))
                if is_graph_out:
                    # A declared output: pass the data through unchanged (Identity). The exact declared
                    # shape is guaranteed by the ORT-driven _coerce_output_shapes post-pass, which uses
                    # authoritative inference (our lighter interpreter's tail shapes can drift).
                    onnx_nodes.append(helper.make_node("Identity", [new_in[0]], new_out, name=nn["name"]))
                    for o, uo in zip(nd["outputs"], new_out):
                        if o:
                            producers.setdefault(o, []).append(uo)
                    continue
                if counts_ok:
                    tname = new_out[0] + "__rtgt"
                    onnx_nodes.append(helper.make_node(
                        "Constant", [], [tname], name=tname + "_c",
                        value=numpy_helper.from_array(np.array(rec_out, dtype=np.int64), name=tname + "_v")))
                    interp.val[tname] = np.array(rec_out, dtype=np.int64)
                    interp.shp[tname] = [len(rec_out)]
                    new_in = [new_in[0], tname]
                elif concrete(dsh0) and tgt_len_c is not None:
                    # computed target we can't trust: pin a geometry-valid target of the SAME rank as
                    # the (collapsed) target, keeping the data's element count.
                    tot = int(np.prod(dsh0)); k = tgt_len_c
                    if k == len(dsh0):
                        newt = list(dsh0)
                    elif k > len(dsh0):
                        newt = [1] * (k - len(dsh0)) + list(dsh0)            # unsqueeze-like
                    else:
                        keep = list(dsh0)[len(dsh0) - (k - 1):] if k > 1 else []
                        newt = [tot // int(np.prod(keep)) if keep else tot] + keep
                    tname = new_out[0] + "__rtgt"
                    onnx_nodes.append(helper.make_node(
                        "Constant", [], [tname], name=tname + "_c",
                        value=numpy_helper.from_array(np.array(newt, dtype=np.int64), name=tname + "_v")))
                    interp.shp[tname] = [k]
                    new_in = [new_in[0], tname]
                else:
                    _fix_shape_control(interp, op, new_in, attrs, const_node, set_const, rec_out, args.faithful)
            else:
                _fix_shape_control(interp, op, new_in, attrs, const_node, set_const, rec_out, args.faithful)
            interp.run(op, new_in, new_out, attrs)
            onnx_nodes.append(helper.make_node(op, new_in, new_out, name=nn["name"],
                                               **_clean_attrs(op, attrs)))
        for o, uo in zip(nd["outputs"], new_out):
            if o:
                producers.setdefault(o, []).append(uo)

    if args.faithful:
        # Second pass, run once the whole trace is interpreted. Order: (1) rebind provably
        # mis-bound edges (collapsed names wired to the wrong producer); (2) a default-guessed
        # Resize target is re-solved from its CONSUMER's broadcast anchor -- the machinery-built
        # grid/ramp operand of the binary op the Resize output feeds fixes the true spatial dims
        # (the grid chains come later in the trace, so the emission pass could not see them);
        # (3) graph-output Reshape targets are re-imposed with the settled ladder; (4) interior
        # pixel-unshuffle Reshape counts absorb lost literal dims; (5) rebind anything the new
        # shapes exposed.
        onnx_by_name = {n.name: n for n in onnx_nodes}
        _repair_misbindings(new_nodes, onnx_by_name, producers, rec_by_out, interp, ilog)
        cons_of = {}
        for r in new_nodes:
            for pos, s in enumerate(r["in"]):
                if s:
                    cons_of.setdefault(s, []).append((r, pos))

        def anchor_shape(uo, depth=0):
            for (r, pos) in cons_of.get(uo, []):
                if r["op"] in ("Cast", "Identity") and depth < 3:
                    s = anchor_shape(r["out"][0], depth + 1)
                    if s:
                        return s
                if r["op"] in ("Add", "Sub", "Mul", "Div") and len(r["in"]) > 1:
                    sib = r["in"][1 - pos]
                    s = interp.gs(sib) if sib else None
                    if concrete(s) and len(s) == 4 and s[2] > 1 and s[3] > 1:
                        return s
            return None

        for nn, fmode in faithful_resizes:
            if fmode not in ("default-sizes", "default-scales"):
                continue
            S = anchor_shape(nn["out"][0])
            xsh = interp.gs(nn["in"][0])
            if S is None or not (concrete(xsh) and len(xsh) == 4):
                continue
            if fmode == "default-sizes":
                _impose_vec(nn["in"][3], [xsh[0], xsh[1], S[2], S[3]], rec_by_out, const_node,
                            set_const, interp, ilog)
                ilog(f"anchor: Resize {nn['name']} -> {[xsh[0], xsh[1], S[2], S[3]]}")
            else:
                scname = nn["in"][2]
                if scname in const_node:
                    dtn, shp = const_node[scname][1], const_node[scname][2]
                    set_const(scname, np.array([1.0, 1.0, S[2] / xsh[2], S[3] / xsh[3]])
                              .reshape(shp if shp else (4,)), dtn)
                    ilog(f"anchor: Resize {nn['name']} scales -> {[1.0, 1.0, S[2]/xsh[2], S[3]/xsh[3]]}")
            _replay_interp(interp, new_nodes)

        for nn, ro in faithful_out_reshapes:
            if not concrete(ro):
                continue
            dsh0 = interp.gs(nn["in"][0])
            if not concrete(dsh0) or int(np.prod(ro)) != int(np.prod(dsh0)):
                # impose anyway: the declared target's leaf back-solve fixes the SHARED constants
                # (Gather indices, /2 divisors) the interior pixel-unshuffle also uses; the
                # count-fix stage then absorbs what is left.
                ilog(f"impose: output Reshape {nn['name']} count-inconsistent for now "
                     f"(declared {ro}, data {dsh0})")
            _impose_vec(nn["in"][1], ro, rec_by_out, const_node, set_const, interp, ilog)
            _replay_interp(interp, new_nodes)

        _fix_interior_reshape_counts(new_nodes, rec_by_out, const_node, set_const, interp, ilog)
        _repair_misbindings(new_nodes, onnx_by_name, producers, rec_by_out, interp, ilog)

    if args.faithful and args.raw_names:
        # Emit under the ORIGINAL dumped names, duplicates and all: the source export reuses
        # tensor names, and the consumer's importer must do its own SSA-rename/binding -- that
        # binding is part of what the artifact exists to exercise. Constant values are the
        # interpreted (self-consistent) ones; initializers keep their real shapes.
        raw_nodes = []
        for i, nd in enumerate(nodes):
            op = nd["op_type"]
            outs = list(nd["outputs"])
            name = f"{nd['name']}__r{i}"
            if op == "Constant":
                uo = f"{outs[0]}__n{i}" if outs and outs[0] else ""
                val = interp.gv(uo)
                if val is not None:
                    raw_nodes.append(helper.make_node(
                        "Constant", [], outs, name=name,
                        value=numpy_helper.from_array(np.asarray(val), name=name + "_v")))
                else:
                    raw_nodes.append(helper.make_node("Constant", [], outs, name=name,
                                                      **_scalar_attrs(nd["attributes"])))
            elif op == "ConstantOfShape":
                dtn, _ = _const_meta(nd)
                nd_attrs = {}
                if dtn:
                    nd_attrs["value"] = numpy_helper.from_array(np.zeros((1,), dtype=NP_OF[dtn]))
                raw_nodes.append(helper.make_node(op, list(nd["inputs"]), outs, name=name, **nd_attrs))
            else:
                raw_nodes.append(helper.make_node(op, list(nd["inputs"]), outs, name=name,
                                                  **_clean_attrs(op, nd["attributes"])))
        raw_inputs = [helper.make_tensor_value_info(it["name"], ELEM[it["dtype"]], it["shape"])
                      for it in d["inputs"]]
        raw_outputs = [helper.make_tensor_value_info(ot["name"], ELEM[ot["dtype"]],
                       ot["shape"] if concrete(ot["shape"]) else None) for ot in d["outputs"]]
        rgraph = helper.make_graph(raw_nodes, "rebuilt_raw", raw_inputs, raw_outputs, onnx_inits)
        # the source file carries shape-inferred value_info for intermediates (one entry per NAME,
        # even though several nodes produce that name) -- reproduce them, they are part of what an
        # importer must survive.
        for t in d["tensors"]:
            if t.get("category") == "value" and concrete(t.get("shape")) and t["dtype"] in ELEM:
                rgraph.value_info.append(
                    helper.make_tensor_value_info(t["name"], ELEM[t["dtype"]], t["shape"]))
        opset = d["model"]["opset_import"].get("ai.onnx", 17)
        rmodel = helper.make_model(rgraph, opset_imports=[helper.make_opsetid("", int(opset))])
        rmodel.ir_version = min(d["model"].get("ir_version", 8), 10)
        onnx.save(rmodel, args.out)
        print(f"wrote {args.out}: {len(raw_nodes)} RAW-NAME nodes "
              f"(SSA-violating like the source export; ORT will not run it)")
        return

    out_unique = {}
    for on in graph_out_names:
        cands = producers.get(on)
        out_unique[on] = cands[-1] if cands else on

    for on in graph_out_names:
        onnx_nodes.append(helper.make_node("Identity", [out_unique[on]], [on], name=f"__out_{on}"))

    # Broadcast-repair (build-time): in the soft-argmax coordinate-decode tail, index-ramp initializers
    # multiply warp tensors whose exact spatial layout is baked and lost, so an elementwise operand may
    # not broadcast. Where the offending operand is a constant/initializer, neutralise it to a
    # broadcastable scalar of the same dtype (op/dtype/structure preserved). Data-operand clashes are
    # left to the ORT-driven _repair_broadcasts post-pass (which uses a Slice, never a Resize).
    if not args.faithful:
        _broadcast_repair(onnx_nodes, onnx_inits, interp, ELEM_INT_TO_NP)

    # Dead-node elimination: keep only nodes on a backward path from the declared outputs. This removes
    # the grid-builder subgraphs whose GridSample grid was replaced -- and, because those replaced grids
    # no longer reference the conv-derived optical flow (the flow tensors' names are lost to the SSA
    # collapse and cannot be re-identified for the final warps), the convolution pyramid is unreachable
    # from the outputs and is dropped here. The kept subgraph exercises the full dynamic-shape tail
    # (Shape/Slice/Concat/Reshape/Resize-geometry, GridSample, ConstantOfShape/Where/Expand, the
    # soft-argmax coordinate decode) and fully determines the declared outputs.
    if not args.faithful:
        onnx_nodes = _prune_dead(onnx_nodes, graph_out_names, init_names | input_names)
    onnx_inits = [ini for ini in onnx_inits if ini.name in _used_names(onnx_nodes)]

    onnx_inputs = [helper.make_tensor_value_info(it["name"], ELEM[it["dtype"]], it["shape"])
                   for it in d["inputs"]]
    onnx_outputs = []
    for ot in d["outputs"]:
        # faithful mode: declare outputs rank-only dynamic -- the runtime machinery determines the
        # shape, and ORT-vs-vknn on the SAME graph is the shape oracle (a static declaration the
        # interior ladder can't guarantee would only block the session).
        sh = ([None] * len(ot["shape"]) if ot["shape"] else None) if args.faithful \
            else (ot["shape"] if concrete(ot["shape"]) else None)
        onnx_outputs.append(helper.make_tensor_value_info(ot["name"], ELEM[ot["dtype"]], sh))

    graph = helper.make_graph(onnx_nodes, "rebuilt", onnx_inputs, onnx_outputs, onnx_inits)
    opset = d["model"]["opset_import"].get("ai.onnx", 17)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", int(opset))])
    model.ir_version = min(d["model"].get("ir_version", 8), 10)

    # Final output-shape guarantee: using ORT's own (authoritative) shape inference -- not our lighter
    # interpreter -- coerce every declared output whose producer's element count drifted from the
    # declared shape. The output-formatting tail (pixel-unshuffle and coordinate-decode
    # branches) depends on a resolution ladder the collapsed dump doesn't preserve, so a branch may
    # arrive at the wrong count/rank. We splice Reshape([1,1,1,-1]) -> Resize(nearest, declared) so the
    # output is exactly the declared shape/dtype; only its resampled content differs (repro contract).
    # Neutralise interior shape ops the collapsed dump left inconsistent -- broken Resizes (a size
    # vector whose length != data rank, from a mis-bound low-rank producer) become Identity, and
    # pixel-unshuffle Reshapes whose target count disagrees with the ORT-inferred data become
    # count-preserving. Both preserve rank flow and never fail; the declared output shape is then set
    # by output coercion.
    # Iterate the repairs: each pass lets ORT shape inference resolve further, exposing the next
    # inconsistency (a broken Resize, a bad pixel-unshuffle Reshape, a non-broadcasting Mul).
    if args.faithful:
        model = _faithful_ort_repair(model, ilog)
    if not args.faithful:
        for _ in range(6):
            model = _repair_resizes(model)
            model = _repair_broadcasts(model)
            model = _repair_reshapes(model)
        # Some tail shapes are data-dependent and invisible to static inference: fix them from RUNTIME
        # shapes -- run ORT, and on a Reshape/Resize/broadcast failure repair that node from the real
        # (executed) input shape, then retry. No Resize is introduced (Reshape/Slice only). Runs BEFORE
        # output coercion so the coercion sees the final (repaired) branch shapes.
        model = _repair_from_runtime(model, d, rng)
        model = _coerce_output_shapes(model, {o["name"]: o["shape"] for o in d["outputs"]},
                                      {o["name"]: ELEM[o["dtype"]] for o in d["outputs"]},
                                      _make_feeds(d, np.random.default_rng(999)))
    if implog:
        seen_msgs = set()
        for m in implog:
            if m not in seen_msgs:
                seen_msgs.add(m)
                print("  [faithful] " + m)

    onnx.save(model, args.out)
    print(f"wrote {args.out}: {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers")

    feeds = _make_feeds(d, rng)
    if args.inputs_dir:
        os.makedirs(args.inputs_dir, exist_ok=True)
        for it in d["inputs"]:
            a = feeds[it["name"]]
            np.save(os.path.join(args.inputs_dir, it["name"] + ".npy"), a)
            a.tofile(os.path.join(args.inputs_dir, it["name"] + ".bin"))
        print(f"wrote inputs -> {args.inputs_dir}")

    if args.check:
        _check(model, d, feeds, args.goldens_dir)


_BINARY_EW = {"Add", "Sub", "Mul", "Div", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
              "Equal", "Min", "Max", "Pow", "And", "Or"}


def _broadcast_repair(nodes, inits, interp, elem_np):
    """Make elementwise binary operands broadcast: where a constant/initializer operand's shape does
    not broadcast with its sibling's, swap it for a broadcastable scalar of the same dtype."""
    init_shape = {ini.name: list(ini.dims) for ini in inits}
    const_out = {}  # name -> (node, dtype_np) for materialised Constant tensors
    for n in nodes:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    const_out[n.output[0]] = (n, numpy_helper.to_array(a.t).dtype)
    scalar_counter = [0]

    def shape_of(name):
        if name in init_shape:
            return init_shape[name]
        return interp.gs(name)

    def dtype_of(name):
        if name in const_out:
            return const_out[name][1]
        for ini in inits:
            if ini.name == name:
                return numpy_helper.to_array(ini).dtype
        return None

    def out_dt_bool(op):
        return op in ("Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Equal", "And", "Or")

    extra = []
    for n in list(nodes):
        if n.op_type not in _BINARY_EW or len(n.input) < 2:
            continue
        a, b = n.input[0], n.input[1]
        sa, sb = shape_of(a), shape_of(b)
        if not (concrete(sa) and concrete(sb)):
            continue
        if _broadcastable(sa, sb) or _broadcastable(sb, sa):
            # keep interp shape current for downstream repairs
            if n.output and not out_dt_bool(n.op_type):
                interp.shp[n.output[0]] = _bcast(sa, sb)
            continue
        # (a) one operand is a constant/initializer -> neutralise it to a broadcastable scalar
        fixed = False
        for pos, name in ((1, b), (0, a)):
            if name in const_out or name in init_shape:
                dt = dtype_of(name)
                if dt is None:
                    continue
                sname = f"__bcastfix_{scalar_counter[0]}"
                scalar_counter[0] += 1
                sc = np.zeros((1,), dtype=dt)
                node = helper.make_node("Constant", [], [sname], name=sname + "_c",
                                        value=numpy_helper.from_array(sc, name=sname + "_v"))
                extra.append(node)
                interp.shp[sname] = [1]
                n.input[pos] = sname
                fixed = True
                break
        if fixed:
            other = shape_of(n.input[0]) if pos == 1 else shape_of(n.input[1])
            if n.output and concrete(other):
                interp.shp[n.output[0]] = list(other)
            continue
        # (b) two non-broadcasting DATA operands: left to the ORT-driven _repair_broadcasts post-pass,
        # which reduces the clashing dims with a Slice (no Resize, so no channel/count change).
    if extra:
        nodes[:0] = extra  # Constants first (no inputs) -> valid topological order


def _runtime_shape(model, tensor, feeds):
    """Return `tensor`'s concrete runtime shape by executing a graph PRUNED to just its cone (so a
    later broken node is never run)."""
    import onnxruntime as ort
    m2 = onnx.ModelProto()
    m2.CopyFrom(model)
    by_out = {o: n for n in m2.graph.node for o in n.output if o}
    keep = set()
    stack = [tensor]
    while stack:
        nm = stack.pop()
        p = by_out.get(nm)
        if p is None or id(p) in keep:
            continue
        keep.add(id(p))
        for i in p.input:
            if i:
                stack.append(i)
    kept = [n for n in m2.graph.node if id(n) in keep]
    del m2.graph.node[:]
    m2.graph.node.extend(kept)
    del m2.graph.output[:]
    vi = onnx.ValueInfoProto()
    vi.name = tensor  # no declared type/shape -> ORT infers it (UNDEFINED elem_type fails to load)
    m2.graph.output.append(vi)
    used = {i for n in m2.graph.node for i in n.input}
    keep_init = [ini for ini in m2.graph.initializer if ini.name in used]
    del m2.graph.initializer[:]
    m2.graph.initializer.extend(keep_init)
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 4  # silence expected probe/repair failures
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(m2.SerializeToString(), so, providers=["CPUExecutionProvider"])
        feed = {k: v for k, v in feeds.items() if k in {i.name for i in m2.graph.input}}
        val = sess.run([tensor], feed)[0]
        return list(val.shape)
    except Exception:
        return None


def _repair_from_runtime(model, d, rng, max_iter=60):
    """Run ORT; on a shape failure at a Reshape/Resize/binary node, repair it with the node's real
    runtime input shape (Reshape target / Slice for broadcast / Resize->Identity), then retry. Also
    probes each declared output's cone so hidden failures (masked by an earlier one) get fixed too."""
    import onnxruntime as ort
    import re as _re
    feeds = _make_feeds(d, np.random.default_rng(12345))
    out_names = [o["name"] for o in d["outputs"]]

    def next_failure():
        # try the whole model, then each output cone, and return the first failing node name
        for target in [None] + out_names:
            try:
                so = ort.SessionOptions()
                so.log_severity_level = 4  # silence expected probe/repair failures
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                if target is None:
                    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
                    sess.run(None, feeds)
                else:
                    _runtime_shape_or_raise(model, target, feeds)
            except Exception as e:
                m = _re.search(r"Node \(([^)]+)\)", str(e)) or _re.search(r"Name:'([^']+)'", str(e))
                if m:
                    return m.group(1)
        return None

    for _ in range(max_iter):
        nname = next_failure()
        if nname is None:
            return model
        node = next((n for n in model.graph.node if n.name == nname), None)
        if node is None or not _repair_node_runtime(model, node, feeds):
            return model
    return model


def _runtime_shape_or_raise(model, tensor, feeds):
    """Like _runtime_shape but raises the ORT error (so the caller can find the failing node)."""
    import onnxruntime as ort
    m2 = onnx.ModelProto()
    m2.CopyFrom(model)
    by_out = {o: n for n in m2.graph.node for o in n.output if o}
    keep, stack = set(), [tensor]
    while stack:
        nm = stack.pop()
        p = by_out.get(nm)
        if p is None or id(p) in keep:
            continue
        keep.add(id(p))
        for i in p.input:
            if i:
                stack.append(i)
    del m2.graph.node[:]
    m2.graph.node.extend([n for n in model.graph.node if id(n) in keep])
    del m2.graph.output[:]
    vi = onnx.ValueInfoProto()
    vi.name = tensor  # no declared type/shape -> ORT infers it (UNDEFINED elem_type fails to load)
    m2.graph.output.append(vi)
    used = {i for n in m2.graph.node for i in n.input}
    keep_init = [ini for ini in m2.graph.initializer if ini.name in used]
    del m2.graph.initializer[:]
    m2.graph.initializer.extend(keep_init)
    so = ort.SessionOptions()
    so.log_severity_level = 4  # silence expected probe/repair failures
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(m2.SerializeToString(), so, providers=["CPUExecutionProvider"])
    sess.run([tensor], {k: v for k, v in feeds.items() if k in {i.name for i in m2.graph.input}})


def _repair_node_runtime(model, node, feeds):
    """Fix one node using runtime input shapes. Returns True if it changed something."""
    by_out = {o: n for n in model.graph.node for o in n.output if o}
    if node.op_type == "Reshape" and len(node.input) >= 2:
        # keep the target RANK (a following Transpose may depend on it); use -1 for the last dim so the
        # element count matches at runtime regardless of the (possibly hidden) data shape.
        k = 4
        tn = by_out.get(node.input[1])
        if tn is not None and tn.op_type == "Constant":
            for a in tn.attribute:
                if a.name == "value":
                    k = len(np.atleast_1d(numpy_helper.to_array(a.t)))
        newt = np.array([1] * (k - 1) + [-1], dtype=np.int64)
        tname = node.name + "__rtfix"
        model.graph.node.insert(0, helper.make_node("Constant", [], [tname], name=tname + "_c",
                                                    value=numpy_helper.from_array(newt, name=tname + "_v")))
        node.input[1] = tname
        return True
    if node.op_type == "Resize":
        # neutralise to Identity (a broken size vector from a collapse artifact)
        node.op_type = "Identity"
        del node.input[1:]
        for a in list(node.attribute):
            node.attribute.remove(a)
        return True
    if node.op_type in _BINARY_EW and len(node.input) >= 2:
        sa = _runtime_shape(model, node.input[0], feeds)
        sb = _runtime_shape(model, node.input[1], feeds)
        if not (concrete(sa) and concrete(sb)) or len(sa) != len(sb):
            return False
        axes = [ax for ax in range(len(sa)) if sa[ax] != sb[ax] and sa[ax] != 1 and sb[ax] != 1]
        if not axes:
            return False
        base = node.name + "__rtb"
        st, en, axc, sln = base + "s", base + "e", base + "a", base + "l"
        for nm, arr in ((st, np.zeros(len(axes), np.int64)), (en, np.ones(len(axes), np.int64)),
                        (axc, np.array(axes, np.int64))):
            model.graph.node.insert(0, helper.make_node("Constant", [], [nm], name=nm + "_c",
                                                        value=numpy_helper.from_array(arr, name=nm + "_v")))
        idx = list(model.graph.node).index(node)
        model.graph.node.insert(idx, helper.make_node("Slice", [node.input[1], st, en, axc], [sln],
                                                      name=sln + "_s"))
        node.input[1] = sln
        return True
    return False


def _repair_broadcasts(model, max_iter=10):
    """Make elementwise binary ops broadcast, using ORT-authoritative inference. Where two operands'
    inferred shapes don't broadcast, Resize the second (rank-4) operand to the first's shape, or
    neutralise a constant/initializer operand to a scalar. Iterated (each fix unlocks inference)."""
    for _ in range(max_iter):
        try:
            inf = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
        except Exception:
            break
        shp = {}  # value: dim list with -1 for symbolic dims, or None if rank unknown
        for vi in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output):
            dims = [dm.dim_value if dm.HasField("dim_value") else -1
                    for dm in vi.type.tensor_type.shape.dim]
            shp[vi.name] = dims if dims else None
        init_names = {ini.name for ini in model.graph.initializer}
        const_names = {n.output[0] for n in model.graph.node if n.op_type == "Constant"}

        def compat(x, y):  # broadcast-compatible dim (treat symbolic -1 as wildcard)
            return x == y or x == 1 or y == 1 or x == -1 or y == -1

        def bcast_ok(sx, sy):
            if sx is None or sy is None:
                return True  # unknown rank -> assume ok
            k = max(len(sx), len(sy))
            px = [1] * (k - len(sx)) + list(sx)
            py = [1] * (k - len(sy)) + list(sy)
            return all(compat(u, v) for u, v in zip(px, py))

        changed = False
        inserts = []
        for idx, n in enumerate(model.graph.node):
            if n.op_type not in _BINARY_EW or len(n.input) < 2:
                continue
            a, b = n.input[0], n.input[1]
            sa, sb = shp.get(a), shp.get(b)
            if sa is None or sb is None:
                continue
            if bcast_ok(sa, sb):
                continue
            # neutralise a constant/initializer operand to a scalar (keeps op/dtype, drops the clash)
            fixed = False
            for pos, name in ((1, b), (0, a)):
                if name in const_names or name in init_names:
                    sc = "%s__bfix%d" % (n.name, idx)
                    inserts.append((idx, helper.make_node(
                        "Constant", [], [sc], name=sc + "_c",
                        value=numpy_helper.from_array(np.ones((1,), dtype=np.float32), name=sc + "_v"))))
                    # match dtype to the sibling via a Cast so types agree
                    cst = sc + "_cast"
                    inserts.append((idx, helper.make_node("CastLike", [sc, n.input[1 - pos]], [cst],
                                                          name=cst + "_cl")))
                    n.input[pos] = cst
                    fixed = True
                    changed = True
                    break
            if fixed:
                continue
            # two rank-N data operands that don't broadcast -> reduce operand b's mismatched dims to 1
            # with a Slice (a real op the model uses), so it broadcasts against a. No Resize -- the
            # element count only shrinks along clashing axes; op/dtype structure is preserved.
            if len(sa) == len(sb):
                axes = [ax for ax in range(len(sa))
                        if not compat(sa[ax], sb[ax])]  # clashing concrete dims only
                if axes:
                    st = "%s__ss%d" % (n.name, idx)
                    en = "%s__se%d" % (n.name, idx)
                    axc = "%s__sa%d" % (n.name, idx)
                    sln = "%s__sl%d" % (n.name, idx)
                    inserts.append((idx, helper.make_node("Constant", [], [st], name=st + "_c",
                        value=numpy_helper.from_array(np.zeros(len(axes), np.int64), name=st + "_v"))))
                    inserts.append((idx, helper.make_node("Constant", [], [en], name=en + "_c",
                        value=numpy_helper.from_array(np.ones(len(axes), np.int64), name=en + "_v"))))
                    inserts.append((idx, helper.make_node("Constant", [], [axc], name=axc + "_c",
                        value=numpy_helper.from_array(np.array(axes, np.int64), name=axc + "_v"))))
                    inserts.append((idx, helper.make_node("Slice", [b, st, en, axc], [sln], name=sln + "_s")))
                    n.input[1] = sln
                    changed = True
        for off, (idx, node) in enumerate(inserts):
            model.graph.node.insert(idx + off, node)
        if not changed:
            break
    return model


def _repair_resizes(model, max_iter=6):
    """Turn any Resize whose `sizes` length disagrees with its data rank into an Identity. Such a
    Resize comes from a size vector built off a mis-bound low-rank producer (an SSA-collapse artifact);
    making it a no-op removes the shape error without changing the reachable op structure."""
    for _ in range(max_iter):
        try:
            inf = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
        except Exception:
            break
        rank = {}
        for vi in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output):
            dims = vi.type.tensor_type.shape.dim
            rank[vi.name] = len(dims) if len(dims) else None
        init_arr = {ini.name: numpy_helper.to_array(ini) for ini in model.graph.initializer}
        by_out = {o: n for n in model.graph.node for o in n.output if o}
        inserts_local = []

        def data_is_lowrank(name, depth=6):
            # a Resize data that traces (through elementwise ops) to a large 1-D arange constant is a
            # mis-bound coordinate axis, not a feature map -> the Resize is a collapse artifact.
            cur = name
            for _ in range(depth):
                p = by_out.get(cur)
                if p is None:
                    return False
                if p.op_type == "Constant":
                    for a in p.attribute:
                        if a.name == "value":
                            v = numpy_helper.to_array(a.t)
                            return v.ndim == 1 and v.size >= 16
                    return False
                if p.op_type in ("Mul", "Add", "Sub", "Div", "Reciprocal", "Neg", "Cast"):
                    # follow the operand that is itself 1-D-ish (the coordinate), i.e. any input
                    for inp in p.input:
                        if inp and data_is_lowrank(inp, depth - 1):
                            return True
                    return False
                return False
            return False

        changed = False
        for n in model.graph.node:
            if n.op_type != "Resize" or len(n.input) < 4 or not n.input[3]:
                continue
            drank = rank.get(n.input[0])
            szv = None
            if n.input[3] in init_arr:
                szv = init_arr[n.input[3]]
            else:
                sn = by_out.get(n.input[3])
                if sn is not None and sn.op_type == "Constant":
                    for a in sn.attribute:
                        if a.name == "value":
                            szv = numpy_helper.to_array(a.t)
            dsh = None
            for vi in list(inf.graph.value_info) + list(inf.graph.input):
                if vi.name == n.input[0]:
                    dd = [d.dim_value if d.HasField("dim_value") else None
                          for d in vi.type.tensor_type.shape.dim]
                    dsh = dd if all(x is not None for x in dd) else None
            broken = (drank is not None and drank < 4) or data_is_lowrank(n.input[0])
            len_mismatch = szv is not None and drank is not None and len(np.atleast_1d(szv)) != drank
            computed_sizes = szv is None
            if broken:
                n.op_type = "Identity"
                del n.input[1:]
                for a in list(n.attribute):
                    n.attribute.remove(a)
                changed = True
            elif (len_mismatch or computed_sizes) and concrete(dsh) and len(dsh) == 4:
                # replace the (wrong/computed) size vector with an identity [N,C,H,W] constant
                szname = n.name + "__fixsz"
                model.graph.node  # ensure attr access
                sz = numpy_helper.from_array(np.array(dsh, dtype=np.int64), name=szname + "_v")
                cnode = helper.make_node("Constant", [], [szname], name=szname + "_c", value=sz)
                inserts_local.append(cnode)
                while len(n.input) < 4:
                    n.input.append("")
                n.input[1] = ""
                n.input[2] = ""
                n.input[3] = szname
                changed = True
        if inserts_local:
            model.graph.node.extend(inserts_local)  # Constants (no inputs) -> valid anywhere
        if not changed:
            break
    return model


def _repair_reshapes(model, max_iter=8):
    """Iteratively make every interior Reshape valid: where the target's element count disagrees with
    the data's ORT-inferred count, replace the target with the data's own inferred shape (identity).
    Iterated because fixing one reshape lets inference resolve shapes further downstream."""
    for _ in range(max_iter):
        try:
            inf = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
        except Exception:
            break
        shp = {}
        for vi in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output):
            dims, ok = [], True
            for dm in vi.type.tensor_type.shape.dim:
                if dm.HasField("dim_value"):
                    dims.append(dm.dim_value)
                else:
                    ok = False
                    break
            shp[vi.name] = dims if ok else None
        init_arr = {ini.name: numpy_helper.to_array(ini) for ini in model.graph.initializer}
        by_out = {o: n for n in model.graph.node for o in n.output if o}
        changed = False
        inserts = []  # (index, constant_node)
        for idx, n in enumerate(model.graph.node):
            if n.op_type != "Reshape" or len(n.input) < 2:
                continue
            dsh = shp.get(n.input[0])
            if not concrete(dsh):
                continue
            # resolve the target vector value (constant node or initializer)
            tval = None
            if n.input[1] in init_arr:
                tval = init_arr[n.input[1]]
            else:
                tn = by_out.get(n.input[1])
                if tn is not None and tn.op_type == "Constant":
                    for a in tn.attribute:
                        if a.name == "value":
                            tval = numpy_helper.to_array(a.t)
            if tval is None:
                continue
            tval = np.atleast_1d(tval).astype(np.int64)
            if (tval > 0).all() and int(np.prod(tval)) == int(np.prod(dsh)):
                continue  # already valid
            # Preserve the TARGET RANK (a following Transpose's perm depends on it) but make the count
            # match the data: put the full count in the last dim, 1s elsewhere.
            k = len(tval)
            tot = int(np.prod(dsh))
            newt = np.array([1] * (k - 1) + [tot], dtype=np.int64)
            tname = n.name + "__idtgt"
            inserts.append((idx, helper.make_node("Constant", [], [tname], name=tname + "_c",
                                                   value=numpy_helper.from_array(newt, name=tname + "_v"))))
            n.input[1] = tname
            changed = True
        for off, (idx, cn) in enumerate(inserts):
            model.graph.node.insert(idx + off, cn)  # Constant just before its consuming Reshape
        if not changed:
            break
    return model


def _coerce_output_shapes(model, out_shapes, out_elems, feeds=None):
    """Guarantee each declared output has its declared shape, using ORT-authoritative shape inference.

    For an output whose producer's inferred element count differs from the declared count (or whose
    shape isn't fully inferable), rename the producer's output and splice:
        producer -> Reshape([1,1,1,-1]) -> Resize(nearest, [declared]) -> Reshape([declared]) = <out>
    so the graph output is exactly `[declared]` at the declared dtype. Idempotent-safe: only outputs
    that need it are touched.
    """
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception:
        inferred = model
    shp = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
        dims = []
        ok = True
        for dm in vi.type.tensor_type.shape.dim:
            if dm.HasField("dim_value"):
                dims.append(dm.dim_value)
            else:
                ok = False
                break
        shp[vi.name] = dims if ok else None

    g = model.graph
    by_out = {}
    for n in g.node:
        for o in n.output:
            if o:
                by_out[o] = n
    extra = []
    for oname, oshape in out_shapes.items():
        if not concrete(oshape):
            continue
        prod = by_out.get(oname)
        if prod is None:
            continue
        # Use the shape actually FLOWING INTO the output tensor. The output value_info is pre-set to
        # the declared shape, so it can't be trusted; the producer's real data shape can. For an
        # Identity/Reshape producer, that is its data input; otherwise fall back to the tensor itself.
        # rename the producer's output to an internal name, then measure the RUNTIME shape flowing in
        # (the tail is data-dependent, so static inference and the pre-set output value_info both lie).
        internal = oname + "__pre"
        for i, o in enumerate(prod.output):
            if o == oname:
                prod.output[i] = internal
        cur = None
        if feeds is not None:
            cur = _runtime_shape(model, internal, feeds)
        if not concrete(cur):
            cur = shp.get(internal)   # NOT shp.get(oname): that is the lying pre-set output value_info
        if concrete(cur) and list(cur) == list(oshape):
            # already exactly the declared shape -> restore the direct output (undo rename)
            for i, o in enumerate(prod.output):
                if o == internal:
                    prod.output[i] = oname
            continue
        tgt_t = numpy_helper.from_array(np.array(oshape, dtype=np.int64), name=oname + "__tgtt")
        want = int(np.prod(oshape))
        have = int(np.prod(cur)) if concrete(cur) else want
        if have == want:
            # element count already matches -> a single real Reshape to the declared shape (this is
            # exactly the model's true final reshape, e.g. a [1,H,W,2] -> [1,1,H,2W] plane repack).
            extra += [
                helper.make_node("Constant", [], [oname + "__tgt"], name=oname + "__tgt_c", value=tgt_t),
                helper.make_node("Reshape", [internal, oname + "__tgt"], [oname], name=oname + "__final_r"),
            ]
        elif want % have == 0:
            # count drifted by an integer factor (a pixel-unshuffle output branch arrives at half the
            # channel count because the exact upstream resolution isn't recoverable). Bridge it with a
            # channel Concat (a real op the model uses) -- flatten, Concat `factor` copies, Reshape.
            factor = want // have
            flat_t = numpy_helper.from_array(np.array([1, -1], dtype=np.int64), name=oname + "__flatt")
            extra.append(helper.make_node("Constant", [], [oname + "__flat"], name=oname + "__flat_c", value=flat_t))
            extra.append(helper.make_node("Reshape", [internal, oname + "__flat"], [oname + "__r2"], name=oname + "__r2_r"))
            extra.append(helper.make_node("Concat", [oname + "__r2"] * factor, [oname + "__cat"],
                                          name=oname + "__cat_c", axis=1))
            extra += [
                helper.make_node("Constant", [], [oname + "__tgt"], name=oname + "__tgt_c", value=tgt_t),
                helper.make_node("Reshape", [oname + "__cat", oname + "__tgt"], [oname], name=oname + "__final_r"),
            ]
        else:
            # non-integer drift (rare): flatten and slice/pad-free Reshape to a count-matching shape,
            # then Reshape to declared. Uses only Reshape (equal count each step).
            flat_t = numpy_helper.from_array(np.array([-1], dtype=np.int64), name=oname + "__flatt")
            extra.append(helper.make_node("Constant", [], [oname + "__flat"], name=oname + "__flat_c", value=flat_t))
            extra.append(helper.make_node("Reshape", [internal, oname + "__flat"], [oname + "__r1"], name=oname + "__r1_r"))
            extra += [
                helper.make_node("Constant", [], [oname + "__tgt"], name=oname + "__tgt_c", value=tgt_t),
                helper.make_node("Reshape", [oname + "__r1", oname + "__tgt"], [oname], name=oname + "__final_r"),
            ]
    if extra:
        g.node.extend(extra)
    return model


def _used_names(nodes):
    names = set()
    for n in nodes:
        names.update(n.input)
    return names


def _prune_dead(nodes, graph_outputs, roots):
    """Keep only nodes on a backward path from the declared graph outputs."""
    by_out = {}
    for n in nodes:
        for o in n.output:
            if o:
                by_out[o] = n
    keep = set()
    stack = list(graph_outputs)
    while stack:
        name = stack.pop()
        prod = by_out.get(name)
        if prod is None or id(prod) in keep:
            continue
        keep.add(id(prod))
        for inp in prod.input:
            if inp:
                stack.append(inp)
    return [n for n in nodes if id(n) in keep]


def _make_feeds(d, rng):
    feeds = {}
    for it in d["inputs"]:
        npdt = NP_OF[it["dtype"]]; shp = it["shape"]; n = int(np.prod(shp))
        if it["dtype"] == "UINT8":
            feeds[it["name"]] = rng.integers(0, 256, size=n, dtype=np.uint8).reshape(shp)
        elif it["dtype"] in ("FLOAT16", "FLOAT", "DOUBLE"):
            # bounded [0,1): any float input (blend alpha, normalized planes) stays in a range the
            # synthesized constants keep fp16-finite; a standard-normal alpha (negative / >1) is not a
            # value the real pipeline ever feeds.
            feeds[it["name"]] = rng.uniform(0.0, 1.0, n).astype(npdt).reshape(shp)
        else:
            feeds[it["name"]] = np.zeros(n, dtype=npdt).reshape(shp)
    return feeds


def _scalar_attrs(attrs):
    return {k: v for k, v in attrs.items() if k != "value"}


def _clean_attrs(op, attrs):
    out = {}
    for k, v in attrs.items():
        if isinstance(v, str) and v.startswith("<"):
            continue
        out[k] = v
    return out


def _check(model, d, feeds, goldens_dir):
    import onnxruntime as ort
    try:
        onnx.checker.check_model(model, full_check=False)
        print("onnx.checker: OK")
    except Exception as e:
        print("onnx.checker FAILED:", str(e)[:400])
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
        out_names = [x["name"] for x in d["outputs"]]
        outs = sess.run(out_names, feeds)
        print("ORT run: OK")
        if goldens_dir:
            os.makedirs(goldens_dir, exist_ok=True)
        for o, name in zip(outs, out_names):
            print(f"  ORT out {name}: {list(o.shape)} {o.dtype}")
            if goldens_dir:
                np.save(os.path.join(goldens_dir, name + ".npy"), o)
                o.tofile(os.path.join(goldens_dir, name + ".bin"))
        if goldens_dir:
            print(f"wrote goldens -> {goldens_dir}")
    except Exception as e:
        print("ORT run FAILED:", str(e)[:800])
        sys.exit(3)


if __name__ == "__main__":
    main()
