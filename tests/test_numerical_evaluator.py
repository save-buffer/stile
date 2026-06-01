"""
Tests for the FP-rounded evaluator over the raw `ExprType` tree.

The headline behaviors:

  1. Each AST op attaches one rounding-noise symbol with coefficient
     ≤ ε · |value|.
  2. Operation order matters — `(a + b) + c` and `a + (b + c)`
     produce *different* AffineForms (different op-count and
     partial-sum trajectories), even though both compute the same
     mathematical value.
  3. Two trees that share a tensor leaf share its noise symbol, so
     subtracting their AffineForms cancels the leaf-noise component
     and leaves only the per-op rounding gap.
  4. `HardwareNumerics.reduction_order` swaps the reduction-error
     formula at evaluation time without re-walking the tree.
"""
import math

import pytest

import stile.numerical as nx
from stile import dim
from stile.numerical import (
    AffineForm, fresh_noise, evaluate, exprs_close,
    HardwareNumerics, WORST_CASE, NVIDIA_TENSOR_CORE_TF32, TPU_MXU,
    MACHINE_EPS,
)
from stile.type import Constant, Tensor, BinaryOp, UnaryOp, Reduce


def _leaf(name : str, central : float, radius : float) -> AffineForm:
    """
    Convenience: leaf AffineForm with one noise symbol at the
    given central value + radius. Pinned name so tests sharing this
    helper across two ETs share the noise symbol via the leaves dict
    (the evaluator looks tensors up by name).
    """
    return AffineForm.with_noise(central, fresh_noise(name), radius)


def test_leaf_lookup_round_trips(reset):
    """A bare Tensor evaluates to whatever the leaves dict says."""
    N = dim("EvN1", 4)
    x_form = _leaf("X", central=2.0, radius=0.5)
    et = Tensor(dims=(N,), name="X")
    result = evaluate(et, {"X" : x_form})
    assert result.central == 2.0
    assert result.range() == (1.5, 2.5)


def test_constant_has_no_noise(reset):
    """A bare Constant has zero radius and no noise."""
    et = Constant(value=3.14)
    result = evaluate(et, leaf_forms={})
    assert result.central == 3.14
    assert result.total_radius() == 0.0


def test_binary_op_adds_rounding_noise(reset):
    """An `X + X` BinaryOp should grow the radius by at most ε·|X|."""
    N = dim("EvN2", 4)
    x_form = _leaf("X", central=1.0, radius=0.1)
    et = BinaryOp(op="+", lhs=Tensor(dims=(N,), name="X"),
                          rhs=Tensor(dims=(N,), name="X"))
    result = evaluate(et, {"X" : x_form})
    # Central: 2 * 1 = 2. Radius before rounding: 2 * 0.1 = 0.2.
    # Rounding adds at most ε · max|2x| = ε · 2.2.
    eps = MACHINE_EPS["float32"]
    expected_radius = 0.2 + eps * 2.2
    assert math.isclose(result.central, 2.0, rel_tol=1e-9)
    assert math.isclose(result.total_radius(), expected_radius, rel_tol=1e-9)


def test_operation_order_matters(reset):
    """
    `(a + b) + c` and `a + (b + c)` are *structurally distinct* ETs.
    Both compute the same math; both produce AffineForms whose central
    values match exactly (linear adds are exact in AA central
    arithmetic); both attach two rounding-noise symbols (one per `+`).
    Their AA *forms* differ in which intermediate value gets rounded
    — left-assoc rounds `a+b` first, right-assoc rounds `b+c` first.
    """
    N = dim("OrdN", 4)
    a = _leaf("A", 1.0, 0.1)
    b = _leaf("B", 2.0, 0.1)
    c = _leaf("C", 3.0, 0.1)
    leaves = {"A" : a, "B" : b, "C" : c}

    t_a = Tensor(dims=(N,), name="A")
    t_b = Tensor(dims=(N,), name="B")
    t_c = Tensor(dims=(N,), name="C")
    left  = BinaryOp(op="+", lhs=BinaryOp(op="+", lhs=t_a, rhs=t_b), rhs=t_c)
    right = BinaryOp(op="+", lhs=t_a, rhs=BinaryOp(op="+", lhs=t_b, rhs=t_c))

    e_left = evaluate(left, leaves)
    e_right = evaluate(right, leaves)
    # Central matches (linear).
    assert math.isclose(e_left.central, e_right.central, rel_tol=1e-9)
    # Each tree has two `+` ops, so two new rounding noises per tree.
    # Leaf-noise components are identical (same A/B/C). The fresh
    # round-noises differ — so the noise tuples aren't equal.
    assert e_left.noise != e_right.noise


def test_exprs_close_shares_leaf_noises(reset):
    """
    Two ETs that reference the same leaf share its noise symbol, so
    `exprs_close` subtracts and the leaf-noise cancels exactly. The
    remaining gap is only the rounding-op gap.
    """
    N = dim("ShN", 4)
    a = _leaf("A", 1.0, 0.5)
    b = _leaf("B", 2.0, 0.5)
    leaves = {"A" : a, "B" : b}

    t_a = Tensor(dims=(N,), name="A")
    t_b = Tensor(dims=(N,), name="B")
    left  = BinaryOp(op="+", lhs=BinaryOp(op="+", lhs=t_a, rhs=t_b), rhs=t_a)
    right = BinaryOp(op="+", lhs=t_a, rhs=BinaryOp(op="+", lhs=t_b, rhs=t_a))

    close, diff = exprs_close(
        left, right, leaf_forms=leaves, tolerance=1e-3,
    )
    # Mathematical value is the same; the diff's central is exactly 0
    # (linear add is exact). Leaf noises cancel. Only fresh
    # rounding noises remain.
    assert diff.central == 0.0
    # The diff range is bounded by O(ε · max|value|), well within 1e-3.
    assert close
    # The leaf noises don't appear in `diff` because they shared
    # symbols and cancelled.
    leaf_symbols_in_diff = [
        s for s, _ in diff.noise if s.label in ("A", "B")
    ]
    assert leaf_symbols_in_diff == []


def test_reduce_sum_sequential_grows_quadratically(reset):
    """
    Sequential reduction's error bound: `ε · max|x| · N(N-1)/2`.
    For N=10 and max|x|=1, fp32 ε ≈ 6e-8, so bound ≈ 6e-8 · 45 ≈ 2.7e-6.
    """
    N = dim("RedN", 10)
    x = _leaf("X", 1.0, 0.0)   # zero radius → all rounding error
    et = Reduce(op="sum", dim=N, child=Tensor(dims=(N,), name="X"))
    result = evaluate(et, {"X" : x}, hardware=WORST_CASE)
    # Central = 10 (sum of 10 copies of 1).
    assert math.isclose(result.central, 10.0, rel_tol=1e-9)
    # Radius bound: ε · 1 · 45 plus a final round-fp at the call site.
    eps = MACHINE_EPS["float32"]
    expected = eps * 1.0 * (10 * 9 // 2)
    assert math.isclose(result.total_radius(), expected, rel_tol=1e-6)


def test_reduce_tree_has_log_depth_error(reset):
    """
    Tree reduction's error scales with `N · log₂ N`, much tighter
    than sequential's `N²/2`.
    """
    N = dim("RedTN", 1024)
    x = _leaf("X", 1.0, 0.0)
    et = Reduce(op="sum", dim=N, child=Tensor(dims=(N,), name="X"))
    seq_hw = HardwareNumerics(
        name="seq", reduction_order="sequential",
        default_dtype="float32",
    )
    tree_hw = HardwareNumerics(
        name="tree", reduction_order="tree",
        default_dtype="float32",
    )
    r_seq = evaluate(et, {"X" : x}, hardware=seq_hw)
    r_tree = evaluate(et, {"X" : x}, hardware=tree_hw)
    # Tree should be MUCH tighter at N=1024.
    # Seq: ε · 1 · 1024 · 1023 / 2 ≈ ε · 5.2e5
    # Tree: ε · 1 · 1024 · 10        ≈ ε · 1.0e4
    # Ratio ≈ 50×.
    assert r_tree.total_radius() < 0.1 * r_seq.total_radius()


def test_hardware_model_accumulator_dtype_overrides_default(reset):
    """
    On a backend with bf16 inputs but fp32 accumulator (TPU/Trainium),
    the reduction error should reflect fp32 ε, not bf16 ε.
    """
    N = dim("AccDtN", 100)
    x = _leaf("X", 1.0, 0.0)
    et = Reduce(op="sum", dim=N, child=Tensor(dims=(N,), name="X"))
    bf16_in_bf16_acc = HardwareNumerics(
        name="bf16-acc", default_dtype="bfloat16",
        accumulator_dtype="bfloat16", reduction_order="sequential",
    )
    bf16_in_fp32_acc = HardwareNumerics(
        name="bf16-in-fp32-acc", default_dtype="bfloat16",
        accumulator_dtype="float32", reduction_order="sequential",
    )
    r_bf16 = evaluate(et, {"X" : x}, hardware=bf16_in_bf16_acc)
    r_mixed = evaluate(et, {"X" : x}, hardware=bf16_in_fp32_acc)
    # bf16 ε ≈ 4e-3, fp32 ε ≈ 6e-8 — accumulator dtype shrinks the
    # reduction-error coefficient by a factor of ~65000.
    assert r_mixed.total_radius() < 1e-3 * r_bf16.total_radius()


def test_unknown_dtype_raises(reset):
    """The HardwareNumerics constructor rejects unknown dtypes."""
    with pytest.raises(ValueError, match="unknown default_dtype"):
        HardwareNumerics(default_dtype="float7")


def test_multiplicative_mask_evaluates_to_piecewise_noise(reset):
    """
    A `Cond(P, 1, 0)` mask evaluates to AffineForm(0.5, ((ε_mask, 0.5),))
    — mid = 0.5, half_spread = 0.5, with a tag-noise that controls
    whether the cell selects the on or off branch. Range = [0, 1].
    """
    from stile.type import TagCond
    from stile.indexing import domain, lt, LoopVariable
    N = dim("MaskN1", 4)
    v = LoopVariable("MaskN1")
    P = domain([v], [lt(v, 2)])
    mask = Tensor(
        dims=(N,),
        tag=TagCond(P, Constant(1.0), Constant(0.0)),
        name="_mask",
    )
    result = evaluate(mask, leaf_forms={})
    assert result.central == 0.5
    assert result.range() == (0.0, 1.0)
    assert len(result.noise) == 1


def test_mask_minus_mask_cancels(reset):
    """
    Two references to the same mask in a difference cancel exactly
    because the tag-noise cache assigns them the same ε_mask. This
    is the AA analog of stile's structural `mask - mask = 0` rule
    (`test_mask_minus_mask_is_zero` in test_tagged_tensors.py).
    """
    from stile.type import TagCond, BinaryOp
    from stile.indexing import domain, lt, LoopVariable
    N = dim("MaskN2", 4)
    v = LoopVariable("MaskN2")
    P = domain([v], [lt(v, 2)])
    mask = Tensor(
        dims=(N,),
        tag=TagCond(P, Constant(1.0), Constant(0.0)),
        name="_mask",
    )
    diff = BinaryOp(op="-", lhs=mask, rhs=mask)
    result = evaluate(diff, leaf_forms={})
    # Diff central = 0; mask term cancels through the shared cache;
    # only the `-`'s rounding noise remains (bounded by ε · |value|;
    # mask value is at most 1, so error ≤ ε).
    assert result.central == 0.0
    eps = MACHINE_EPS["float32"]
    assert result.total_radius() <= eps * 1.0 + 1e-12


def test_bias_mask_via_exp_short_circuits(reset):
    """
    `exp(x + Cond(P, 0, -inf))` short-circuits into a piecewise form
    bounded by [0, exp(x_max)]. Without the special case, the bare
    `Cond(P, 0, -inf)` evaluation would raise (half_spread is ∞).
    """
    from stile.type import TagCond, BinaryOp, UnaryOp
    from stile.indexing import domain, lt, LoopVariable
    N = dim("BiasN", 4)
    v = LoopVariable("BiasN")
    P = domain([v], [lt(v, 2)])
    bias_mask = Tensor(
        dims=(N,),
        tag=TagCond(P, Constant(0.0), Constant(float("-inf"))),
        name="_bmask",
    )
    x = _leaf("X", central=0.0, radius=1.0)   # x ∈ [-1, 1]
    et = UnaryOp(
        op="exp",
        child=BinaryOp(op="+", lhs=Tensor(dims=(N,), name="X"), rhs=bias_mask),
    )
    result = evaluate(et, leaf_forms={"X" : x})
    # Bound: 0 (from ¬P) ∪ exp(x.range()) = [0, exp(1)] ≈ [0, 2.72].
    lo, hi = result.range()
    assert lo <= 0.0 < math.exp(1.0) <= hi


def test_bare_bias_mask_raises_with_hint(reset):
    """A bias mask outside `exp(... + mask)` raises with a pointer."""
    from stile.type import TagCond
    from stile.indexing import domain, lt, LoopVariable
    N = dim("BiasN2", 4)
    v = LoopVariable("BiasN2")
    P = domain([v], [lt(v, 2)])
    bias = Tensor(
        dims=(N,),
        tag=TagCond(P, Constant(0.0), Constant(float("-inf"))),
        name="_bmask",
    )
    with pytest.raises(NotImplementedError, match="bias mask"):
        evaluate(bias, leaf_forms={})


def test_gather_passes_source_aa_through(reset):
    """
    `Gather(source, dim, idx)` returns source's AffineForm — any
    output cell is a valid source value, and the noise symbols share
    so two gathers of the same source still cancel correlations.
    """
    from stile.type import Gather
    N = dim("GatN", 4)
    M = dim("GatM", 4)
    src = _leaf("S", central=3.0, radius=0.5)
    idx_form = _leaf("idx", central=0.0, radius=2.0)
    et = Gather(
        source=Tensor(dims=(N,), name="S"),
        dim_in_source=N,
        idx=Tensor(dims=(M,), name="idx"),
    )
    result = evaluate(et, {"S" : src, "idx" : idx_form})
    # Should be identical to S's AA form (gather doesn't widen).
    assert result.central == 3.0
    assert result.range() == (2.5, 3.5)
