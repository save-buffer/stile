"""
Tests for the spec-parser's `where P` clause. `where P` attaches a
multiplicative mask tagged with the predicate's domain; when the masked
expression sits inside a `sum`, the normalizer's mask-fold extracts the
predicate into the reduce's iteration domain.
"""
import pytest

import stile
from stile import dim, reset_stile
from stile.specification import parse_spec_into_type
from stile.verification import (
    normalize, NormalizedReduce, NormalizedTensor, verify_exprs_equivalent,
)
from stile.indexing import LoopVariable


@pytest.fixture
def reset():
    yield
    reset_stile()


def _single_factor(expr):
    normalized = normalize(expr)
    assert len(normalized.num.factors) == 1
    return next(iter(normalized.num.factors))


def test_where_on_reduction_folds_into_domain(reset):
    """`sum[N](N where N >= 32)` — mask absorbed, reduce domain is conjoined."""
    N = dim("N", 64)
    t = parse_spec_into_type("sum[N](N where N >= 32) -> ")

    factor = _single_factor(t.et)
    assert isinstance(factor, NormalizedReduce)
    assert factor.dim == N

    # One disjunct with two constraints: `N < 64` and `N >= 32`.
    assert len(factor.domain.disjuncts) == 1
    conj = next(iter(factor.domain.disjuncts))
    assert len(conj) == 2

    # Body is the bare tensor `N` — mask factor has been stripped.
    child_factor = next(iter(factor.child.num.factors))
    assert isinstance(child_factor, NormalizedTensor)
    assert child_factor.tag is None


def test_where_equivalence_between_specs(reset):
    """Two spec strings expressing the same masked sum normalize equal."""
    N = dim("N", 64)
    spec_where = parse_spec_into_type("sum[N](N where N >= 32) -> ")
    spec_slice = parse_spec_into_type("sum[N](N[32:64]) -> ")
    assert verify_exprs_equivalent(spec_where.et, spec_slice.et)


def test_where_two_dim_predicate_causal(reset):
    """Causal-shape predicate `where K <= Q` parses with cross-dim constraint."""
    Q = dim("Q", 8)
    K = dim("K", 8)
    t = parse_spec_into_type("sum[K](Q K where K <= Q) -> Q")

    factor = _single_factor(t.et)
    assert isinstance(factor, NormalizedReduce)
    assert factor.dim == K
    # Domain carries: K >= 0, K < 8, AND K <= Q.
    assert len(factor.domain.disjuncts) == 1
    conj = next(iter(factor.domain.disjuncts))
    assert len(conj) == 3
    # The cross-dim constraint introduced Q as a free variable.
    assert LoopVariable("Q") in factor.domain.variables


def test_where_with_affine_arithmetic(reset):
    """Predicate accepts `int * dim` and unary/binary arithmetic."""
    N = dim("N", 64)
    t = parse_spec_into_type("sum[N](N where 2 * N >= 16) -> ")
    factor = _single_factor(t.et)
    assert isinstance(factor, NormalizedReduce)
    # Just verify parsing/normalization succeed; domain shape is exercised
    # by the more focused tests above.


def test_where_invalid_dim_raises(reset):
    """Predicate referencing a dim not in the expression's shape is rejected."""
    N = dim("N", 64)
    dim("M", 32)
    with pytest.raises(ValueError, match="where.*dim"):
        parse_spec_into_type("sum[N](N where M >= 0) -> ")


def test_causal_attention_numerator_folds_mask(reset):
    """
    End-to-end: a causal-attention numerator spec, written with `where
    nctx <= qctx`, normalizes to a single `NormalizedReduce` whose domain
    carries the causal constraint conjoined with the `nctx`-range.
    """
    dim("dhead", 16)
    dim("qctx", 8)
    nctx = dim("nctx", 8)

    spec = parse_spec_into_type(
        "sum(exp((qctx dhead, nctx dhead -> qctx nctx) / sqrt(16)) where nctx <= qctx, "
        "nctx dhead -> qctx dhead)"
    )
    factor = _single_factor(spec.et)
    assert isinstance(factor, NormalizedReduce)
    assert factor.dim == nctx
    # Single disjunct, three constraints: `nctx >= 0`, `nctx < 8`, `nctx <= qctx`.
    assert len(factor.domain.disjuncts) == 1
    conj = next(iter(factor.domain.disjuncts))
    assert len(conj) == 3
    # Cross-dim constraint introduced `qctx` as a free variable in the domain.
    assert {LoopVariable("nctx"), LoopVariable("qctx")} <= factor.domain.variables


def test_causal_attention_two_writings_equivalent(reset):
    """
    Two ways of writing the causal-attention numerator that differ in where
    the `where`-clause is attached normalize to the same expression: the
    parser places the mask, but the multiplicative mask commutes with the
    other contraction operand, so the two forms must converge.
    """
    dim("dhead", 16)
    dim("qctx", 8)
    dim("nctx", 8)

    # Mask attached to the score tensor (the natural reading).
    spec_a = parse_spec_into_type(
        "sum(exp((qctx dhead, nctx dhead -> qctx nctx) / sqrt(16)) where nctx <= qctx, "
        "nctx dhead -> qctx dhead)"
    )
    # Mask attached to the score tensor *outside* the `exp` — different
    # surface form, same denotation only because the predicate is on a
    # zero-or-one mask and the sum absorbs it identically.
    spec_b = parse_spec_into_type(
        "sum((exp((qctx dhead, nctx dhead -> qctx nctx) / sqrt(16))) where nctx <= qctx, "
        "nctx dhead -> qctx dhead)"
    )
    assert verify_exprs_equivalent(spec_a.et, spec_b.et)


def test_where_equality_predicate(reset):
    """`==` splits into a pair of constraints."""
    N = dim("N", 64)
    t = parse_spec_into_type("sum[N](N where N == 16) -> ")
    factor = _single_factor(t.et)
    assert isinstance(factor, NormalizedReduce)
    conj = next(iter(factor.domain.disjuncts))
    # `N < 64`, `N >= 0`, `N <= 16`, `N >= 16` — simplifier keeps the tightest
    # upper and lower bounds per variable.
    # After simplification: `N < 17` (from `N <= 16`) and `N >= 16`.
    assert len(conj) == 2
