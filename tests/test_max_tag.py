"""
Max-side tag handling — the counterpart of #7 for max reductions.

  - `Reduce(max, child=Tensor(tag=Cond(D, x, -inf)))` extracts to a
    max over `x` with iteration domain conjoined with `D` (already
    handled by `_extract_tagged_body` for `op="max"`).

  - `score + Cond(D, 0, -inf)` distributes `+` through the tag and the
    `make_sum` `-inf` absorption produces `Cond(D, score, -inf)`,
    which the max extractor then folds. Bias-form ≡ explicit-mask
    convergence on the max side.

  - Adjacent max-Reduces over sliced sub-ranges sharing a body and a
    cross-variable predicate (`extras`) fuse into a single max-Reduce
    over the union of intervals — mirror of `_merge_sum_reduces` for
    max via `_split_reduce_domain` + `_rebuild_reduce_with_extras`.
"""
import math
import pytest

from stile import dim, reset_stile
from stile.type import Tensor, Constant, TagCond, BinaryOp, Reduce
from stile.indexing import LoopVariable, domain, le
from stile.verification import verify_exprs_equivalent


@pytest.fixture
def reset():
    yield
    reset_stile()


def _bias_tag(N, Q, n, q):
    return Tensor(
        dims=(N, Q),
        tag=TagCond(
            domain=domain([n, q], [le(n, q)]),
            if_true=Constant(0.0),
            if_false=Constant(float("-inf")),
        ),
    )


def _explicit_tag(N, Q, n, q, score):
    return Tensor(
        dims=(N, Q),
        tag=TagCond(
            domain=domain([n, q], [le(n, q)]),
            if_true=score,
            if_false=Constant(float("-inf")),
        ),
    )


def test_max_of_score_plus_bias_equals_explicit_tag(reset):
    """`max_N(score + Cond(D, 0, -inf))` ≡ `max_N(Cond(D, score, -inf))`."""
    N = dim("MTN", 8)
    Q = dim("MTQ", 8)
    n = LoopVariable("MTN")
    q = LoopVariable("MTQ")
    score = Tensor(dims=(N, Q))

    biased = Reduce(
        op="max", dim=N,
        child=BinaryOp("+", score, _bias_tag(N, Q, n, q)),
    )
    explicit = Reduce(op="max", dim=N, child=_explicit_tag(N, Q, n, q, score))
    assert verify_exprs_equivalent(biased, explicit)


def test_two_tile_max_fuses_with_shared_predicate(reset):
    """`max(max_{n in [0,4)} (score+bias), max_{n in [4,8)} (score+bias))`
    ≡ `max_{n in [0,8)} (score+bias)`. The bias→tag fold runs per-tile,
    then `make_max`'s reduce-merge unions the intervals while preserving
    the shared cross-variable predicate."""
    N = dim("MFN", 8)
    Q = dim("MFQ", 8)
    n = LoopVariable("MFN")
    q = LoopVariable("MFQ")
    score = Tensor(dims=(N, Q))
    bias = _bias_tag(N, Q, n, q)

    def tile_max(s, e):
        return Reduce(
            op="max",
            dim=N[s:e],
            child=BinaryOp("+", score, bias),
        )

    two_tile = BinaryOp("max", tile_max(0, 4), tile_max(4, 8))
    single = Reduce(op="max", dim=N, child=BinaryOp("+", score, bias))
    assert verify_exprs_equivalent(two_tile, single)


def test_max_drops_neg_inf_child(reset):
    """`max(a, -inf)` collapses to `a` — pre-existing behavior, pinned
    here so future regressions in `make_max` are caught alongside the
    new tagged-fusion work."""
    N = dim("MDN", 4)
    a = Tensor(dims=(N,))
    expr = BinaryOp("max", a, Constant(float("-inf")))
    assert verify_exprs_equivalent(expr, a)
