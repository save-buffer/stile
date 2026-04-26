"""
Mixing tagged and untagged partial sums.

When a kernel splits a sum into tile contributions, sub-diagonal tiles
need no mask (the predicate holds throughout the tile range), while
on-diagonal tiles do need it. The natural way to write this is
"untagged sub-diagonal contribution + tagged diagonal contribution"
— but the verifier currently treats them as different normalized
factors and won't fuse them.

The implicit-zero-bias rule: when adding a tagged sum-reduce to an
untagged sum-reduce that share the same body and dim, treat the
untagged side as having the *same* predicate (an implicit "always
satisfied" tag). The combined result has the merged interval and
the shared predicate. Mathematically valid because the kernel author
arranged the tile ranges so the untagged tile's interval lies fully
inside the predicate's satisfying set.
"""
import pytest

from stile import dim, reset_stile
from stile.type import Tensor, Constant, TagCond, Reduce, BinaryOp
from stile.indexing import LoopVariable, domain, le, ge
from stile.verification import (
    normalize, verify_exprs_equivalent, NormalizedReduce,
)


@pytest.fixture
def reset():
    yield
    reset_stile()


def _masked_tile_sum(N : "FullDim", body, start : int, end : int, predicate_domain):
    """`sum_{k in [start,end)} body * Cond(predicate, 1, 0)`."""
    sliced_dim = N[start:end]
    mask = Tensor(
        dims=(N,),
        tag=TagCond(predicate_domain, Constant(1.0), Constant(0.0)),
    )
    return Reduce(
        op="sum",
        dim=sliced_dim,
        child=BinaryOp(op="*", lhs=body, rhs=mask),
    )


def _plain_tile_sum(N : "FullDim", body, start : int, end : int):
    """`sum_{k in [start,end)} body` — no mask."""
    return Reduce(op="sum", dim=N[start:end], child=body)


@pytest.mark.xfail(
    reason="Sound implicit-zero-bias requires polyhedral subsumption "
    "against the outer slice range of `q`, which lives in `Type.st` and "
    "isn't visible to the post-hoc verifier. Tracked as task #6, "
    "blocked on task #8."
)
def test_tagged_plus_untagged_collapses_to_tagged_full(reset):
    """
    `(sum_{k in [0,4)} f(k)) + (sum_{k in [4,8) where k<=q} f(k))`
    should normalize to `sum_{k in [0,8) where k<=q} f(k)` *when* the
    surrounding context constrains `q` to a range where the untagged
    tile is in-mask — e.g., the kernel sliced `q` to `[3, 8)`, so for
    every `q` the untagged range `[0, 4)` satisfies `k <= q`. Today
    that slice info isn't threaded into the verifier, so the promotion
    can't be soundly fired.
    """
    N = dim("UM_N", 16)
    Q = dim("UM_Q", 16)
    body = Tensor(dims=(N, Q))

    n_var = LoopVariable("UM_N")
    q_var = LoopVariable("UM_Q")
    causal = domain([n_var, q_var], [le(n_var, q_var)])

    untagged = _plain_tile_sum(N, body, 0, 4)
    tagged = _masked_tile_sum(N, body, 4, 8, causal)
    kernel = BinaryOp(op="+", lhs=untagged, rhs=tagged)

    spec = _masked_tile_sum(N, body, 0, 8, causal)
    assert verify_exprs_equivalent(kernel, spec)


@pytest.mark.xfail(
    reason="Same architectural gap as the two-way case. Task #6, "
    "blocked on task #8."
)
def test_three_way_mix_collapses(reset):
    """
    Three contiguous tiles, the first untagged and the rest tagged with
    the same predicate, should collapse to a single tagged sum over the
    full range.
    """
    N = dim("UM3_N", 12)
    Q = dim("UM3_Q", 12)
    body = Tensor(dims=(N, Q))

    n_var = LoopVariable("UM3_N")
    q_var = LoopVariable("UM3_Q")
    causal = domain([n_var, q_var], [le(n_var, q_var)])

    a = _plain_tile_sum(N, body, 0, 4)
    b = _masked_tile_sum(N, body, 4, 8, causal)
    c = _masked_tile_sum(N, body, 8, 12, causal)
    kernel = BinaryOp(op="+", lhs=BinaryOp(op="+", lhs=a, rhs=b), rhs=c)

    spec = _masked_tile_sum(N, body, 0, 12, causal)
    assert verify_exprs_equivalent(kernel, spec)


def test_two_tagged_tiles_already_fuse(reset):
    """Sanity baseline: two tagged tiles with the same predicate already
    fuse via `_merge_sum_reduces` (landed in the previous turn). This
    test pins that behavior so a future regression in the tagged-fusion
    path is caught alongside the new tagged+untagged work."""
    N = dim("UM2_N", 16)
    Q = dim("UM2_Q", 16)
    body = Tensor(dims=(N, Q))
    n_var = LoopVariable("UM2_N")
    q_var = LoopVariable("UM2_Q")
    causal = domain([n_var, q_var], [le(n_var, q_var)])

    a = _masked_tile_sum(N, body, 0, 4, causal)
    b = _masked_tile_sum(N, body, 4, 8, causal)
    kernel = BinaryOp(op="+", lhs=a, rhs=b)

    spec = _masked_tile_sum(N, body, 0, 8, causal)
    assert verify_exprs_equivalent(kernel, spec)
