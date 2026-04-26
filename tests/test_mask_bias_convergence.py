"""
The two equivalent ways to mask a softmax-style sum:

  (multiplicative) `exp(x) * Cond(D, 1, 0)`
  (bias-form)      `exp(x + Cond(D, 0, -inf))`

Mathematically both equal `Cond(D, exp(x), 0)`. The kernel side picks
between them based on whether it can afford a separate max pass:

  - Multiplicative is cheap to express but breaks online softmax — the
    masked-out positions become `0`, and `max` will see those zeros
    rather than ignoring them.
  - Bias-form is what every numerically-stable masked online softmax
    actually does, since `+(-inf)` is the identity-on-mask that survives
    the `max - logsumexp` machinery.

This file pins down what the normalizer should do for each form, and
documents the remaining gap so kernel verification has a clear target.
"""
import math
import pytest

from stile import dim, reset_stile
from stile.type import Tensor, Constant, TagCond, BinaryOp, UnaryOp
from stile.indexing import LoopVariable, domain, le
from stile.verification import (
    normalize, verify_exprs_equivalent, NormalizedTensor, NormalizedTagCond,
)


@pytest.fixture
def reset():
    yield
    reset_stile()


def _multiplicative(score_tensor, mask_domain):
    return BinaryOp(
        op="*",
        lhs=UnaryOp(op="exp", child=score_tensor),
        rhs=Tensor(
            dims=score_tensor.dims,
            tag=TagCond(
                domain=mask_domain,
                if_true=Constant(1.0),
                if_false=Constant(0.0),
            ),
        ),
    )


def _bias_form(score_tensor, mask_domain):
    bias = Tensor(
        dims=score_tensor.dims,
        tag=TagCond(
            domain=mask_domain,
            if_true=Constant(0.0),
            if_false=Constant(float("-inf")),
        ),
    )
    return UnaryOp(op="exp", child=BinaryOp(op="+", lhs=score_tensor, rhs=bias))


def test_multiplicative_form_distributes_into_tag(reset):
    """Multiplicative mask: `exp(x) * Cond(D, 1, 0)` collapses to a single
    tagged tensor whose `if_true = exp(x)` and `if_false = 0`."""
    N = dim("MaskN", 8)
    v = LoopVariable("MaskN")
    score = Tensor(dims=(N,))
    mask_domain = domain([v], [le(v, 4)])

    n = normalize(_multiplicative(score, mask_domain))
    assert len(n.num.factors) == 1
    factor = next(iter(n.num.factors))
    assert isinstance(factor, NormalizedTensor)
    assert isinstance(factor.tag, NormalizedTagCond)
    # `if_false` collapsed to the scalar 0; `if_true` is exp(score).
    assert factor.tag.if_false.num.const == 0.0
    assert not factor.tag.if_false.num.factors


def test_bias_form_converges_to_multiplicative(reset):
    """Mathematically `exp(x + Cond(D, 0, -inf)) == exp(x) * Cond(D, 1, 0)`.
    Both forms now land on the same `Cond(D, exp(x), 0)` shape: `+`
    distribution through the tag turns the bias into `Cond(D, x, x-inf)`,
    `make_sum`'s `-inf` absorption folds `x-inf` to `-inf`, and
    `normalize_exp`'s pure-constant fold turns `exp(-inf)` into `0`."""
    N = dim("BiasN", 8)
    v = LoopVariable("BiasN")
    score = Tensor(dims=(N,))
    mask_domain = domain([v], [le(v, 4)])
    assert verify_exprs_equivalent(
        _multiplicative(score, mask_domain),
        _bias_form(score, mask_domain),
    )
