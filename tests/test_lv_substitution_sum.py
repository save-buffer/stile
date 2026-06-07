"""
Regression tests for loop-variable substitution into a `NormalizedSum`.

`substitute_lv_in_expr` used to wedge `NormalizedExpr`s into a
`NormalizedSum`'s children (which must be `NormalizedProduct`s), so even
substituting an *absent* loop variable into `a + b` produced a
structurally-different result that no longer compared equal to the
original. That broke the no-op invariant substitution relies on. These
tests pin the fixed behavior.
"""
from stile.verification import (
    NormalizedTensor, NormalizedProduct, NormalizedExpr, NormalizedSum,
    make_sum, substitute_lv_in_expr,
)
from stile.indexing import LoopVariable
from stile.type import FullDim
from stile.frozen_counter import FrozenCounter


def _sum_expr(*names : str) -> NormalizedExpr:
    N = FullDim("LVN", 8)
    terms = [
        NormalizedProduct(
            factors=FrozenCounter.from_iterable(
                [NormalizedTensor(dims=(N,), tag=None, name=n)]
            )
        )
        for n in names
    ]
    return NormalizedExpr.of(make_sum(terms))


def test_substituting_absent_loop_var_is_noop(reset):
    """`a + b` substituted at an absent loop var must equal itself."""
    expr = _sum_expr("a", "b")
    result = substitute_lv_in_expr(expr, LoopVariable("k"), 0)
    assert result == expr


def test_sum_children_stay_products_after_substitution(reset):
    """The substituted sum's children must remain `NormalizedProduct`s."""
    expr = _sum_expr("a", "b")
    result = substitute_lv_in_expr(expr, LoopVariable("k"), 0)
    the_sum = next(iter(result.num.factors))
    assert isinstance(the_sum, NormalizedSum)
    assert all(isinstance(c, NormalizedProduct) for c in the_sum.children)
