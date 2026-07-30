"""Pass registry: one file per pass, uniform interface, fixed pipeline order.

Order rationale: constants are materialized and folded first so the movement
passes see constant-resolved shapes; movement simplification runs next (the
priority target for vknn -- every removed Reshape/Transpose is GPU layout
churn avoided); dedup/CSE run after simplification exposes duplicates; DCE
runs last to sweep everything the earlier passes orphaned. The engine repeats
the whole pipeline to a fixpoint, so cross-pass cascades resolve without any
pass needing to be clever about ordering.
"""
from onnx_optimizer.passes.base import Pass  # noqa: F401  (re-export)
from onnx_optimizer.passes.cancel_squeeze_unsqueeze import CancelSqueezeUnsqueeze
from onnx_optimizer.passes.constant_to_initializer import ConstantToInitializer
from onnx_optimizer.passes.dedup_initializers import DedupInitializers
from onnx_optimizer.passes.eliminate_common_subexpr import EliminateCommonSubexpr
from onnx_optimizer.passes.eliminate_dead import EliminateDead
from onnx_optimizer.passes.fold_constants import FoldConstants
from onnx_optimizer.passes.fold_conv_batchnorm import FoldConvBatchnorm
from onnx_optimizer.passes.fold_shape_ops import FoldShapeOps
from onnx_optimizer.passes.merge_reshapes import MergeReshapes
from onnx_optimizer.passes.merge_transposes import MergeTransposes
from onnx_optimizer.passes.prune_unused_inputs import PruneUnusedInputs
from onnx_optimizer.passes.remove_dropout import RemoveDropout
from onnx_optimizer.passes.remove_identity import RemoveIdentity
from onnx_optimizer.passes.remove_noop_cast import RemoveNoopCast
from onnx_optimizer.passes.remove_noop_pad import RemoveNoopPad
from onnx_optimizer.passes.remove_noop_reshape import RemoveNoopReshape
from onnx_optimizer.passes.remove_noop_slice import RemoveNoopSlice
from onnx_optimizer.passes.remove_noop_transpose import RemoveNoopTranspose
from onnx_optimizer.passes.remove_single_concat import RemoveSingleConcat

# Pipeline order. Lossy passes are listed at their natural pipeline position
# but are filtered out unless --allow-lossy is given.
PIPELINE = [
    ConstantToInitializer,
    FoldShapeOps,
    FoldConstants,
    FoldConvBatchnorm,      # lossy; skipped by default
    RemoveIdentity,
    RemoveDropout,
    RemoveNoopCast,
    RemoveNoopTranspose,
    MergeTransposes,
    MergeReshapes,
    RemoveNoopReshape,
    CancelSqueezeUnsqueeze,
    RemoveNoopSlice,
    RemoveNoopPad,
    RemoveSingleConcat,
    EliminateCommonSubexpr,
    DedupInitializers,
    PruneUnusedInputs,
    EliminateDead,
]


def build_pipeline(only=None, disable=None, allow_lossy=False, **pass_kwargs):
    """Instantiate the pipeline, honoring --only-pass / --disable-pass /
    --allow-lossy. `pass_kwargs` may carry per-pass constructor options
    (currently max_fold_bytes for fold-constants). Raises ValueError on an
    unknown pass name."""
    known = {cls.name: cls for cls in PIPELINE}
    for name in list(only or []) + list(disable or []):
        if name not in known:
            raise ValueError("unknown pass '%s' (see --list-passes)" % name)
    passes = []
    for cls in PIPELINE:
        if only and cls.name not in only:
            continue
        if disable and cls.name in disable:
            continue
        if cls.lossy and not allow_lossy:
            continue
        if cls is FoldConstants and "max_fold_bytes" in pass_kwargs:
            passes.append(cls(max_fold_bytes=pass_kwargs["max_fold_bytes"]))
        else:
            passes.append(cls())
    return passes
