"""Optimization report: JSON (machine) + human-readable rendering.

One report object accumulates everything the run produced: before/after graph
stats, the per-iteration pass log (applied / no-op / auto-reverted), the final
verification result, and the optional --target vknn analysis.
"""
import json

from onnx_optimizer import graph_util as gu


def model_stats(model, path=None):
    graph = model.graph
    init_bytes = 0
    for t in graph.initializer:
        n = 1
        for d in t.dims:
            n *= int(d)
        init_bytes += n * {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2,
                           11: 8, 12: 4, 13: 8, 16: 2}.get(t.data_type, 4)
    return {
        "path": path,
        "serialized_bytes": model.ByteSize(),
        "ir_version": model.ir_version,
        "opsets": {imp.domain or "ai.onnx": imp.version for imp in model.opset_import},
        "nodes": sum(gu.op_histogram(graph).values()),
        "nodes_by_op": dict(sorted(gu.op_histogram(graph).items())),
        "inputs": len(graph.input),
        "initializers": len(graph.initializer),
        "initializer_bytes": init_bytes,
    }


class Report:
    def __init__(self):
        self.tool = {}
        self.before = {}
        self.after = {}
        self.passes = []        # {iteration, pass, changes, status, detail?}
        self.reverted = []      # pass names auto-reverted by the gate
        self.iterations = 0
        self.verification = None
        self.vknn = None
        self.lossy = None       # tolerance summary when --allow-lossy was used

    def log_pass(self, iteration, name, changes, status, detail=None):
        entry = {"iteration": iteration, "pass": name, "changes": changes, "status": status}
        if detail:
            entry["detail"] = detail
        self.passes.append(entry)
        if status == "reverted" and name not in self.reverted:
            self.reverted.append(name)

    def to_dict(self):
        return {
            "tool": self.tool,
            "model_before": self.before,
            "model_after": self.after,
            "iterations": self.iterations,
            "passes": self.passes,
            "auto_reverted_passes": self.reverted,
            "verification": self.verification,
            "lossy": self.lossy,
            "vknn": self.vknn,
        }

    def write_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def render_text(self):
        lines = []
        b, a = self.before, self.after
        lines.append("== onnx_optimizer report ==")
        if b and a:
            lines.append("nodes:        %d -> %d" % (b["nodes"], a["nodes"]))
            lines.append("initializers: %d (%.2f MB) -> %d (%.2f MB)"
                         % (b["initializers"], b["initializer_bytes"] / 1e6,
                            a["initializers"], a["initializer_bytes"] / 1e6))
            lines.append("model size:   %.2f MB -> %.2f MB"
                         % (b["serialized_bytes"] / 1e6, a["serialized_bytes"] / 1e6))
            ops = sorted(set(b["nodes_by_op"]) | set(a["nodes_by_op"]))
            deltas = []
            for op in ops:
                nb, na = b["nodes_by_op"].get(op, 0), a["nodes_by_op"].get(op, 0)
                if nb != na:
                    deltas.append("  %-24s %4d -> %d" % (op, nb, na))
            if deltas:
                lines.append("op deltas:")
                lines.extend(deltas)
        applied = [p for p in self.passes if p["status"] == "applied"]
        if applied:
            lines.append("passes applied (%d iterations):" % self.iterations)
            for p in applied:
                lines.append("  iter %d  %-26s %d change(s)"
                             % (p["iteration"], p["pass"], p["changes"]))
        if self.reverted:
            lines.append("passes AUTO-REVERTED by the bit-exact gate:")
            for p in self.passes:
                if p["status"] == "reverted":
                    lines.append("  iter %d  %-26s %s"
                                 % (p["iteration"], p["pass"], p.get("detail", "")))
        if self.verification is not None:
            v = self.verification
            n_ok = sum(1 for c in v["cases"] if c["ok"])
            lines.append("verification: %s (%d/%d cases byte-identical; %d skipped)"
                         % ("PASS" if v["ok"] else "FAIL", n_ok, len(v["cases"]),
                            len(v.get("skipped", []))))
        if self.lossy is not None:
            lines.append("LOSSY MODE: byte-equality waived for lossy passes; "
                         "max ULP %s, max abs diff %s"
                         % (self.lossy.get("max_ulp"), self.lossy.get("max_abs_diff")))
        if self.vknn is not None:
            for line in self.vknn.get("text", []):
                lines.append(line)
        return "\n".join(lines)
