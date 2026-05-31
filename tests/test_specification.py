"""
Spec-language parser (`parse_spec_into_type`) behaviors: shaped
constants, loop-variable scoping, and rejection of malformed tensor
references. The `where P` and `[d where P]` clauses have their own
files (test_where_clause, test_dim_annotation_predicate); this covers
the parser's core grammar.
"""
import pytest

from stile import dim
import stile.type as st
from stile.type import Constant
from stile.specification import parse_spec_into_type


# --- loop variables: honored in `where`, rejected in expression position

def test_loop_var_in_where_clause_works(reset):
    """Control: a loop var in a `where`-clause slice bound is honored
    (becomes a SymbolicInt) — the supported path."""
    dim("N", 16)
    t = parse_spec_into_type("sum[N where N < k](X:N)", loop_vars={"k"})
    # Upper bound is the symbolic k.
    assert isinstance(t.et, st.Reduce)


def test_loop_var_in_expression_position_raises(reset):
    """A loop var used as a value in the main expression raises a clear
    error instead of silently becoming an anonymous tensor."""
    dim("N", 16)
    with pytest.raises(ValueError, match="loop variable 'k' used in expression position"):
        parse_spec_into_type("X:N * k", loop_vars={"k"})
    with pytest.raises(ValueError, match="loop variable 'k' used in expression position"):
        parse_spec_into_type("k * X:N", loop_vars={"k"})


def test_loop_var_in_reduce_body_raises_clearly(reset):
    """Inside a reduce body, a loop-var value now gets the clear
    loop-var error rather than a confusing `) expected`."""
    dim("N", 16)
    with pytest.raises(ValueError, match="loop variable 'k' used in expression position"):
        parse_spec_into_type("sum[N](k * X:N)", loop_vars={"k"})


def test_labeled_tensor_sharing_loop_var_name_still_parses(reset):
    """A `k:N` labeled tensor is NOT a loop-var-value misuse — the `:`
    disambiguates, so it still parses as a tensor."""
    dim("N", 16)
    # `k:N` is a tensor named k of shape N; the loop-var guard only fires
    # when the name is used bare (no `:dims`).
    t = parse_spec_into_type("k:N * X:N", loop_vars={"k"})
    assert isinstance(t.et, st.BinaryOp)


# --- shaped constants ---------------------------------------------------

def test_zeros_shaped_constant(reset):
    """`zeros[M N]` is a Constant(0.0) carrying shape (M, N) — the
    spellable form of a flash-attention / tiled-matmul accumulator's
    zero base case."""
    M = dim("M", 8)
    N = dim("N", 4)
    t = parse_spec_into_type("zeros[M N]")
    assert t.st == (M, N)
    assert t.et == Constant(0.0)


def test_full_shaped_constant(reset):
    """`full[M N](c)` is a shaped Constant(c), with negative values
    supported."""
    M = dim("M", 8)
    N = dim("N", 4)
    pos = parse_spec_into_type("full[M N](2.5)")
    assert pos.st == (M, N) and pos.et == Constant(2.5)
    negc = parse_spec_into_type("full[N](-1)")
    assert negc.st == (N,) and negc.et == Constant(-1.0)


def test_shaped_constant_composes_in_expression(reset):
    """A shaped constant works as an operand — e.g. `X:N + zeros[N]`."""
    N = dim("N", 4)
    t = parse_spec_into_type("X:N + zeros[N]")
    assert t.st == (N,)
    assert isinstance(t.et, st.BinaryOp)


# --- malformed tensor references ----------------------------------------

def test_bare_tensor_name_without_dims_rejected(reset):
    """`X*X:N` — the first `X` has no shape annotation. Previously this
    silently collapsed to a lone anonymous rank-0 tensor (dropping
    `*X:N`); now it raises, pointing at the fix."""
    dim("N", 4)
    with pytest.raises(ValueError, match="has no shape annotation"):
        parse_spec_into_type("X*X:N")


def test_squaring_a_tensor_needs_both_operands_spelled(reset):
    """The correct spelling — both operands annotated — parses fine."""
    N = dim("N", 4)
    t = parse_spec_into_type("X:N * X:N")
    assert isinstance(t.et, st.BinaryOp)


def test_anonymous_shaped_tensor_still_allowed(reset):
    """A bare *registered dim* used as a shape (no label) is still a
    legit anonymous tensor — the rejection only fires when nothing was
    consumed (unknown bare name)."""
    N = dim("N", 4)
    t = parse_spec_into_type("N")
    assert t.st == (N,)
    assert isinstance(t.et, st.Tensor)
