"""Per-pass unit tests: a tiny graph exhibiting each pattern, an assertion
that the pattern is removed, and a bit-exact check -- plus negative tests for
graphs a pass must NOT touch (strict mode)."""
import numpy as np
from onnx import TensorProto, helper

from onnx_optimizer.tests.helpers import (assert_bitexact, count_ops, init,
                                          make_model, optimize, vi)

F = TensorProto.FLOAT


def test_remove_identity():
    m = make_model(
        [helper.make_node("Identity", ["x"], ["a"]),
         helper.make_node("Relu", ["a"], ["y"])],
        [vi("x", [2, 3])], [vi("y", [2, 3])])
    opt, _ = optimize(m, only=["remove-identity"])
    assert count_ops(opt, "Identity") == 0
    assert_bitexact(m, opt)


def test_remove_identity_feeding_graph_output():
    m = make_model(
        [helper.make_node("Relu", ["x"], ["a"]),
         helper.make_node("Identity", ["a"], ["y"])],
        [vi("x", [2, 3])], [vi("y", [2, 3])])
    opt, _ = optimize(m, only=["remove-identity"])
    assert count_ops(opt, "Identity") == 0
    assert [o.name for o in opt.graph.output] == ["y"]
    assert_bitexact(m, opt)


def test_remove_dropout_inference():
    m = make_model(
        [helper.make_node("Relu", ["x"], ["r"]),
         helper.make_node("Dropout", ["r", "ratio", "training"], ["y", "mask"])],
        [vi("x", [2, 3])], [vi("y", [2, 3])],
        [init("ratio", np.float32(0.5)), init("training", np.bool_(False))])
    opt, _ = optimize(m, only=["remove-dropout"])
    assert count_ops(opt, "Dropout") == 0
    assert_bitexact(m, opt)


def test_dropout_directly_bridging_input_to_output_survives():
    # Degenerate but legal: Dropout is the ONLY node between a graph input and
    # a graph output. There is no producer to rename, so the bypass must
    # refuse and the node must stay (a graph output needs a producer).
    m = make_model(
        [helper.make_node("Dropout", ["x"], ["y"])],
        [vi("x", [2, 3])], [vi("y", [2, 3])])
    opt, _ = optimize(m, only=["remove-dropout"])
    assert count_ops(opt, "Dropout") == 1
    assert_bitexact(m, opt)


def test_dropout_training_mode_survives():
    # Constant-true training_mode: the pass must not fire. Invoked directly
    # (not via the engine) because a training-mode Dropout is nondeterministic
    # and the engine's verifier correctly refuses to gate such a model.
    from onnx_optimizer.passes.remove_dropout import RemoveDropout
    m = make_model(
        [helper.make_node("Relu", ["x"], ["r"]),
         helper.make_node("Dropout", ["r", "ratio", "training"], ["y"])],
        [vi("x", [2, 3])], [vi("y", [2, 3])],
        [init("ratio", np.float32(0.5)), init("training", np.bool_(True))])
    assert RemoveDropout().run(m) == 0
    assert count_ops(m, "Dropout") == 1


def test_dropout_consumed_mask_survives():
    m = make_model(
        [helper.make_node("Dropout", ["x"], ["y", "mask"]),
         helper.make_node("Cast", ["mask"], ["z"], to=F)],
        [vi("x", [2, 3])], [vi("y", [2, 3]), vi("z", [2, 3])])
    opt, _ = optimize(m, only=["remove-dropout"])
    assert count_ops(opt, "Dropout") == 1


def test_remove_noop_cast():
    m = make_model(
        [helper.make_node("Cast", ["x"], ["a"], to=F),
         helper.make_node("Relu", ["a"], ["y"])],
        [vi("x", [4])], [vi("y", [4])])
    opt, _ = optimize(m, only=["remove-noop-cast"])
    assert count_ops(opt, "Cast") == 0
    assert_bitexact(m, opt)


def test_precision_roundtrip_cast_chain_survives_strict_mode():
    # fp32 -> fp16 -> fp32 quantizes the mantissa; NOT an identity.
    m = make_model(
        [helper.make_node("Cast", ["x"], ["h"], to=TensorProto.FLOAT16),
         helper.make_node("Cast", ["h"], ["y"], to=F)],
        [vi("x", [8])], [vi("y", [8])])
    opt, _ = optimize(m)  # full default pipeline
    assert count_ops(opt, "Cast") == 2
    assert_bitexact(m, opt)


def test_remove_noop_transpose():
    m = make_model(
        [helper.make_node("Transpose", ["x"], ["a"], perm=[0, 1, 2]),
         helper.make_node("Relu", ["a"], ["y"])],
        [vi("x", [2, 3, 4])], [vi("y", [2, 3, 4])])
    opt, _ = optimize(m, only=["remove-noop-transpose"])
    assert count_ops(opt, "Transpose") == 0
    assert_bitexact(m, opt)


def test_transpose_inverse_pair_cancels():
    m = make_model(
        [helper.make_node("Transpose", ["x"], ["a"], perm=[0, 2, 1]),
         helper.make_node("Transpose", ["a"], ["b"], perm=[0, 2, 1]),
         helper.make_node("Relu", ["b"], ["y"])],
        [vi("x", [2, 3, 4])], [vi("y", [2, 3, 4])])
    opt, _ = optimize(m, only=["merge-transposes", "dce"])
    assert count_ops(opt, "Transpose") == 0
    assert_bitexact(m, opt)


def test_transpose_non_inverse_pair_survives():
    # Not inverse: must not cancel to nothing. (Composition into ONE transpose
    # is allowed -- it is exact -- but the data movement must remain.)
    m = make_model(
        [helper.make_node("Transpose", ["x"], ["a"], perm=[1, 2, 0]),
         helper.make_node("Transpose", ["a"], ["b"], perm=[1, 2, 0]),
         helper.make_node("Relu", ["b"], ["y"])],
        [vi("x", [2, 3, 4])], [vi("y", [4, 2, 3])])
    opt, _ = optimize(m)
    assert count_ops(opt, "Transpose") >= 1
    assert_bitexact(m, opt)


def test_merge_reshapes():
    m = make_model(
        [helper.make_node("Reshape", ["x", "s1"], ["a"]),
         helper.make_node("Reshape", ["a", "s2"], ["y"])],
        [vi("x", [2, 3, 4])], [vi("y", [4, 6])],
        [init("s1", np.array([6, 4], dtype=np.int64)),
         init("s2", np.array([4, 6], dtype=np.int64))])
    opt, _ = optimize(m, only=["merge-reshapes", "dce"])
    assert count_ops(opt, "Reshape") == 1
    assert_bitexact(m, opt)


def test_merge_reshapes_zero_copy_dim():
    # s2 contains 0 (copy dim from the INTERMEDIATE tensor): a blind merge
    # would copy the wrong dim; the pass must substitute the static shape.
    m = make_model(
        [helper.make_node("Reshape", ["x", "s1"], ["a"]),
         helper.make_node("Reshape", ["a", "s2"], ["y"])],
        [vi("x", [2, 3, 4])], [vi("y", [6, 4])],
        [init("s1", np.array([6, 4], dtype=np.int64)),
         init("s2", np.array([0, 4], dtype=np.int64))])
    opt, _ = optimize(m)
    assert count_ops(opt, "Reshape") == 1
    assert_bitexact(m, opt)


def test_remove_noop_reshape_and_expand():
    m = make_model(
        [helper.make_node("Reshape", ["x", "s"], ["a"]),
         helper.make_node("Expand", ["a", "s"], ["b"]),
         helper.make_node("Relu", ["b"], ["y"])],
        [vi("x", [2, 3])], [vi("y", [2, 3])],
        [init("s", np.array([2, 3], dtype=np.int64))])
    opt, _ = optimize(m, only=["remove-noop-reshape", "dce"])
    assert count_ops(opt, "Reshape") == 0 and count_ops(opt, "Expand") == 0
    assert_bitexact(m, opt)


def test_squeeze_unsqueeze_pair_cancels():
    m = make_model(
        [helper.make_node("Squeeze", ["x", "ax"], ["a"]),
         helper.make_node("Unsqueeze", ["a", "ax"], ["b"]),
         helper.make_node("Relu", ["b"], ["y"])],
        [vi("x", [1, 3, 4])], [vi("y", [1, 3, 4])],
        [init("ax", np.array([0], dtype=np.int64))])
    opt, _ = optimize(m, only=["cancel-squeeze-unsqueeze", "dce"])
    assert count_ops(opt, "Squeeze") == 0 and count_ops(opt, "Unsqueeze") == 0
    assert_bitexact(m, opt)


def test_squeeze_unsqueeze_different_axes_survive():
    m = make_model(
        [helper.make_node("Squeeze", ["x", "ax0"], ["a"]),
         helper.make_node("Unsqueeze", ["a", "ax2"], ["b"]),
         helper.make_node("Relu", ["b"], ["y"])],
        [vi("x", [1, 3, 4])], [vi("y", [3, 4, 1])],
        [init("ax0", np.array([0], dtype=np.int64)),
         init("ax2", np.array([2], dtype=np.int64))])
    opt, _ = optimize(m)
    assert count_ops(opt, "Squeeze") == 1 and count_ops(opt, "Unsqueeze") == 1
    assert_bitexact(m, opt)


def test_remove_noop_slice():
    m = make_model(
        [helper.make_node("Slice", ["x", "st", "en"], ["a"]),
         helper.make_node("Relu", ["a"], ["y"])],
        [vi("x", [2, 3])], [vi("y", [2, 3])],
        [init("st", np.array([0, 0], dtype=np.int64)),
         init("en", np.array([9999, 9999], dtype=np.int64))])
    opt, _ = optimize(m, only=["remove-noop-slice", "dce"])
    assert count_ops(opt, "Slice") == 0
    assert_bitexact(m, opt)


def test_reversing_slice_survives():
    # step -1 keeps the shape but reverses the data: must NOT be removed.
    m = make_model(
        [helper.make_node("Slice", ["x", "st", "en", "ax", "sp"], ["y"])],
        [vi("x", [4])], [vi("y", [4])],
        [init("st", np.array([-1], dtype=np.int64)),
         init("en", np.array([-9223372036854775807], dtype=np.int64)),
         init("ax", np.array([0], dtype=np.int64)),
         init("sp", np.array([-1], dtype=np.int64))])
    opt, _ = optimize(m)
    assert count_ops(opt, "Slice") == 1
    assert_bitexact(m, opt)


def test_remove_noop_pad_and_nonzero_pad_survives():
    m = make_model(
        [helper.make_node("Pad", ["x", "zeros"], ["a"]),
         helper.make_node("Pad", ["a", "nonzero"], ["y"])],
        [vi("x", [2, 3])], [vi("y", [2, 5])],
        [init("zeros", np.zeros(4, dtype=np.int64)),
         init("nonzero", np.array([0, 1, 0, 1], dtype=np.int64))])
    opt, _ = optimize(m, only=["remove-noop-pad", "dce"])
    assert count_ops(opt, "Pad") == 1
    assert_bitexact(m, opt)


def test_remove_single_concat():
    m = make_model(
        [helper.make_node("Concat", ["x"], ["a"], axis=0),
         helper.make_node("Relu", ["a"], ["y"])],
        [vi("x", [2, 3])], [vi("y", [2, 3])])
    opt, _ = optimize(m, only=["remove-single-concat"])
    assert count_ops(opt, "Concat") == 0
    assert_bitexact(m, opt)


def test_cse_merges_duplicates():
    m = make_model(
        [helper.make_node("Sigmoid", ["x"], ["a"]),
         helper.make_node("Sigmoid", ["x"], ["b"]),
         helper.make_node("Add", ["a", "b"], ["y"])],
        [vi("x", [2, 3])], [vi("y", [2, 3])])
    opt, _ = optimize(m, only=["cse", "dce"])
    assert count_ops(opt, "Sigmoid") == 1
    assert_bitexact(m, opt)


def test_cse_keeps_different_attributes():
    m = make_model(
        [helper.make_node("Flatten", ["x"], ["a"], axis=1),
         helper.make_node("Flatten", ["x"], ["b"], axis=2),
         helper.make_node("Shape", ["a"], ["sa"]),
         helper.make_node("Shape", ["b"], ["sb"]),
         helper.make_node("Concat", ["sa", "sb"], ["y"], axis=0)],
        [vi("x", [2, 3, 4])], [vi("y", [4], TensorProto.INT64)])
    opt, _ = optimize(m, only=["cse"])
    assert count_ops(opt, "Flatten") == 2


def test_dedup_initializers():
    m = make_model(
        [helper.make_node("Add", ["x", "w1"], ["a"]),
         helper.make_node("Add", ["a", "w2"], ["y"])],
        [vi("x", [3])], [vi("y", [3])],
        [init("w1", np.array([1, 2, 3], dtype=np.float32)),
         init("w2", np.array([1, 2, 3], dtype=np.float32))])
    opt, _ = optimize(m, only=["dedup-initializers", "dce"])
    assert len(opt.graph.initializer) == 1
    assert_bitexact(m, opt)


def test_dedup_keeps_different_dtype_same_bytes():
    # Same raw bytes, different dtype: int32 ones vs float32 tiny denormals.
    ones_i32 = np.ones(4, dtype=np.int32)
    same_bytes_f32 = ones_i32.view(np.float32).copy()
    m = make_model(
        [helper.make_node("Cast", ["w_i"], ["ci"], to=F),
         helper.make_node("Add", ["ci", "w_f"], ["s"]),
         helper.make_node("Add", ["x", "s"], ["y"])],
        [vi("x", [4])], [vi("y", [4])],
        [init("w_i", ones_i32), init("w_f", same_bytes_f32)])
    opt, _ = optimize(m, only=["dedup-initializers"])
    assert len(opt.graph.initializer) == 2


def test_dce_drops_unreachable_chain():
    m = make_model(
        [helper.make_node("Relu", ["x"], ["y"]),
         helper.make_node("Sigmoid", ["x"], ["dead1"]),
         helper.make_node("Exp", ["dead1"], ["dead2"])],
        [vi("x", [2])], [vi("y", [2])])
    opt, _ = optimize(m, only=["dce"])
    assert len(opt.graph.node) == 1
    assert_bitexact(m, opt)


def test_prune_unused_inputs():
    m = make_model(
        [helper.make_node("Relu", ["x"], ["y"])],
        [vi("x", [2]), vi("unused", [5])], [vi("y", [2])])
    opt, _ = optimize(m, only=["prune-unused-inputs"])
    assert [i.name for i in opt.graph.input] == ["x"]
    assert_bitexact(m, opt)


def test_constant_to_initializer_and_fold():
    c = helper.make_node("Constant", [], ["c"],
                         value=helper.make_tensor("t", F, [2], [1.0, 2.0]))
    m = make_model(
        [c,
         helper.make_node("Mul", ["c", "c"], ["cc"]),
         helper.make_node("Add", ["x", "cc"], ["y"])],
        [vi("x", [2])], [vi("y", [2])])
    opt, _ = optimize(m, only=["constant-to-initializer", "fold-constants", "dce"])
    assert count_ops(opt, "Constant") == 0 and count_ops(opt, "Mul") == 0
    assert count_ops(opt, "Add") == 1
    assert_bitexact(m, opt)


def test_fold_shape_ops_with_dynamic_dim_outside_window():
    # Shape(start=1) over ["batch", 3, 4]: the dynamic dim is OUTSIDE the
    # window, so the fold is exact; full-range Shape must NOT fold.
    m = make_model(
        [helper.make_node("Shape", ["x"], ["s_full"]),
         helper.make_node("Shape", ["x"], ["s_tail"], start=1),
         helper.make_node("Concat", ["s_full", "s_tail"], ["y"], axis=0)],
        [vi("x", ["batch", 3, 4])], [vi("y", [5], TensorProto.INT64)])
    opt, _ = optimize(m, only=["fold-shape-ops"])
    assert count_ops(opt, "Shape") == 1
    assert_bitexact(m, opt)


def test_overridable_initializer_not_folded():
    # An initializer that is ALSO a graph input can be overridden at run time:
    # constant folding through it would bake the default. Must survive.
    m = make_model(
        [helper.make_node("Add", ["x", "w"], ["y"])],
        [vi("x", [2]), vi("w", [2])], [vi("y", [2])],
        [init("w", np.array([1.0, 2.0], dtype=np.float32))])
    opt, _ = optimize(m, only=["fold-constants"])
    assert count_ops(opt, "Add") == 1


def test_fold_conv_batchnorm_only_with_allow_lossy():
    x = vi("x", [1, 2, 4, 4])
    y = vi("y", [1, 2, 4, 4])
    rng = np.random.default_rng(0)
    inits = [init("w", rng.standard_normal((2, 2, 1, 1)).astype(np.float32)),
             init("scale", rng.standard_normal(2).astype(np.float32)),
             init("bias", rng.standard_normal(2).astype(np.float32)),
             init("mean", rng.standard_normal(2).astype(np.float32)),
             init("var", np.abs(rng.standard_normal(2)).astype(np.float32) + 0.5)]
    nodes = [helper.make_node("Conv", ["x", "w"], ["c"]),
             helper.make_node("BatchNormalization",
                              ["c", "scale", "bias", "mean", "var"], ["y"])]
    m = make_model(nodes, [x], [y], inits)
    strict, _ = optimize(m)
    assert count_ops(strict, "BatchNormalization") == 1  # never folded by default
    assert_bitexact(m, strict)
    lossy, _ = optimize(m, only=["fold-conv-batchnorm", "dce"], allow_lossy=True)
    assert count_ops(lossy, "BatchNormalization") == 0
