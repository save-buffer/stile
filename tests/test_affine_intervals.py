"""
Tests that `NormalizedReduce.intervals` handles affine (SymbolicIndex) bounds
correctly: structural equality, adjacency-based merging, and mixed
concrete/symbolic cases.
"""
import pytest

from stile.indexing import LoopVariable, interval_domain
from stile.verification import (
    NormalizedReduce, make_reduce,
    NormalizedTensor, NormalizedExpr, NormalizedProduct,
)
from stile.type import FullDim
from stile.frozen_counter import FrozenCounter
from stile import reset_stile


def _iv(dim : FullDim, *pairs) -> "Domain":
    """Build a 1-D interval Domain over `dim`'s reduction index."""
    return interval_domain(LoopVariable(dim.name), list(pairs))


@pytest.fixture
def reset():
    yield
    reset_stile()


def _leaf_expr(d : FullDim) -> NormalizedExpr:
    return NormalizedExpr.of(NormalizedTensor(dims=frozenset({d})))


def test_adjacent_symbolic_intervals_merge(reset):
    N = FullDim("IntervalN", 128)
    k = LoopVariable("k")
    child = _leaf_expr(N)

    # Two reduces over [k*32, k*32+32) and [k*32+32, k*32+64) should merge.
    r = make_reduce(
        N, "sum",
        [(k * 32, k * 32 + 32), (k * 32 + 32, k * 32 + 64)],
        child,
    )
    assert r.domain == _iv(N, (k * 32, k * 32 + 64))


def test_concrete_and_symbolic_mix(reset):
    N = FullDim("IntervalM", 64)
    k = LoopVariable("k")
    child = _leaf_expr(N)

    # Concrete (0, 32) and symbolic (32, 32 + k*16) are adjacent at 32.
    r = make_reduce(
        N, "sum",
        [(0, 32), (32, 32 + k * 16)],
        child,
    )
    assert r.domain == _iv(N, (0, 32 + k * 16))


def test_non_adjacent_symbolic_intervals_stay_separate(reset):
    N = FullDim("IntervalP", 128)
    k = LoopVariable("k")
    child = _leaf_expr(N)

    # [k*32, k*32+32) and [k*32+64, k*32+96) aren't adjacent — gap at [k*32+32, k*32+64).
    r = make_reduce(
        N, "sum",
        [(k * 32, k * 32 + 32), (k * 32 + 64, k * 32 + 96)],
        child,
    )
    assert r.domain == _iv(N,
        (k * 32, k * 32 + 32),
        (k * 32 + 64, k * 32 + 96),
    )


def test_reduce_equality_is_insensitive_to_interval_insertion_order(reset):
    N = FullDim("IntervalQ", 64)
    k = LoopVariable("k")
    child = _leaf_expr(N)

    r1 = make_reduce(N, "sum", [(0, 16), (k * 16 + 32, k * 16 + 48)], child)
    r2 = make_reduce(N, "sum", [(k * 16 + 32, k * 16 + 48), (0, 16)], child)
    assert r1 == r2
    assert hash(r1) == hash(r2)


def test_affine_simplification_is_used(reset):
    N = FullDim("IntervalR", 128)
    k = LoopVariable("k")
    child = _leaf_expr(N)

    # (k*32 + 32, k*32 + 64) is adjacency-equivalent to ((k+1)*32, (k+1)*32 + 32)
    # after affine canonicalization, so these two reduces must compare equal.
    r1 = make_reduce(N, "sum", [(k * 32 + 32, k * 32 + 64)], child)
    r2 = make_reduce(N, "sum", [((k + 1) * 32, (k + 1) * 32 + 32)], child)
    assert r1 == r2


def test_overlapping_intervals_are_not_merged(reset):
    """
    `sum(a[0:5]) + sum(a[3:7])` must not canonicalize to `sum(a[0:7])` —
    that silently under-counts the overlap `a[3:5]`. Overlapping intervals
    stay separate in the stored frozenset.
    """
    N = FullDim("IntervalS", 16)
    child = _leaf_expr(N)
    r = make_reduce(N, "sum", [(0, 5), (3, 7)], child)
    assert r.domain == _iv(N, (0, 5), (3, 7))

    # Adjacent (not overlapping) should still merge.
    r_adj = make_reduce(N, "sum", [(0, 5), (5, 7)], child)
    assert r_adj.domain == _iv(N, (0, 7))
