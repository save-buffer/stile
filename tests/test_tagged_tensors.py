"""
Tests for value-tagged tensor types: the `Tensor.tag` field and its
normalized counterpart.
"""
import pytest

import stile
from stile import dim, reset_stile, mask_expr, Domain, LoopVariable
from stile.type import Tensor, Constant, TagCond, Reduce, BinaryOp, Repeat
from stile.indexing import domain, le, ge
from stile.verification import (
    normalize, NormalizedTensor, NormalizedTagCond, NormalizedReduce,
    NormalizedExpr, NormalizedProduct,
)


@pytest.fixture
def reset():
    yield
    reset_stile()


def test_tensor_without_tag_unchanged(reset):
    """Default `tag=None` means a tagged-tensor-aware IR is backwards-compat."""
    N = dim("TT_N_plain", 64)
    plain = Tensor(dims=(N,))
    normalized = normalize(plain)
    assert len(normalized.num.factors) == 1
    factor = next(iter(normalized.num.factors))
    assert isinstance(factor, NormalizedTensor)
    assert factor.tag is None


def test_tensor_with_tag_round_trips(reset):
    """`Tensor` with a tag normalizes; the tag preserves structure."""
    N = dim("TT_N_tag", 64)
    tagged = Tensor(
        dims=(N,),
        tag=TagCond(
            domain=stile.range_domain(LoopVariable("TT_N_tag"), 0, 32),
            if_true=Constant(1.0),
            if_false=Constant(0.0),
        ),
    )
    normalized = normalize(tagged)
    factor = next(iter(normalized.num.factors))
    assert isinstance(factor, NormalizedTensor)
    assert isinstance(factor.tag, NormalizedTagCond)
    # Both branches are NormalizedExprs (leaves are bare exprs, not wrappers).
    assert isinstance(factor.tag.if_true, NormalizedExpr)
    assert isinstance(factor.tag.if_false, NormalizedExpr)


def test_identical_cond_branches_collapse(reset):
    """`Cond(D, v, v)` must simplify to `v` during normalization."""
    N = dim("TT_N_coll", 64)
    trivial = Tensor(
        dims=(N,),
        tag=TagCond(
            domain=stile.range_domain(LoopVariable("TT_N_coll"), 0, 32),
            if_true=Constant(5.0),
            if_false=Constant(5.0),
        ),
    )
    normalized = normalize(trivial)
    factor = next(iter(normalized.num.factors))
    # Tag collapsed to the single NormalizedExpr leaf.
    assert isinstance(factor.tag, NormalizedExpr)


def test_mask_expr_builds_tagged_tensor(reset):
    """`stile.mask_expr(dims, domain)` returns a `Tensor` with `Cond(domain, 1, 0)`."""
    i_dim = dim("TT_i", 16)
    j_dim = dim("TT_j", 16)
    i, j = LoopVariable("TT_i"), LoopVariable("TT_j")
    band = domain([i, j], [le(i - j, 2), le(j - i, 2)])
    mask = mask_expr((i_dim, j_dim), band)

    assert isinstance(mask, Tensor)
    assert mask.dims == (i_dim, j_dim)
    assert isinstance(mask.tag, TagCond)
    assert mask.tag.domain == band
    assert mask.tag.if_true == Constant(1.0)
    assert mask.tag.if_false == Constant(0.0)


def test_mask_folds_into_sum_reduce_domain(reset):
    """
    `sum over dim of (body * Mask(P))` normalizes to `sum over (dim ∩ P) of body`
    — the mask factor is extracted from the body and its domain is conjoined
    into the reduce's domain. This is the core spec-vs-kernel convergence
    for local attention.
    """
    N = dim("MF_N", 128)
    k = LoopVariable("MF_N")

    mask_domain_expr = domain([k], [ge(k, 32)])  # "k >= 32"
    mask = mask_expr((N,), mask_domain_expr)
    T = Tensor(dims=(N,))

    # sum_N (T * mask)
    reduce_expr = Reduce(op="sum", dim=N, child=BinaryOp(op="*", lhs=T, rhs=mask))
    normalized = normalize(reduce_expr)

    # Result: a single NormalizedReduce whose body is just T (mask stripped)
    # and whose domain is `{k : 0 <= k < 128} ∩ {k : k >= 32}`.
    assert len(normalized.num.factors) == 1
    factor = next(iter(normalized.num.factors))
    assert isinstance(factor, NormalizedReduce)
    assert factor.op == "sum"
    assert factor.dim == N

    assert len(factor.domain.disjuncts) == 1
    conj = next(iter(factor.domain.disjuncts))
    # Two constraints: `k >= 32` subsumes `k >= 0`; `k < 128` stays.
    assert len(conj) == 2

    # Body is just T — the mask has been stripped.
    child_factor = next(iter(factor.child.num.factors))
    assert isinstance(child_factor, NormalizedTensor)
    assert child_factor.tag is None


def test_repeat_pushes_through_mask_tag(reset):
    """
    `Repeat(N, mask_M)` normalizes to a tagged tensor with dims `{M, N}`
    and the same scalar-valued tag. Scalar leaves broadcast, so the tag
    structure survives the shape extension and the mask stays recognizable
    for downstream push-through and reduce-fold.
    """
    M = dim("RT_M", 32)
    N = dim("RT_N", 64)
    i = LoopVariable("RT_M")

    mask_M = mask_expr((M,), domain([i], [ge(i, 8)]))
    repeated = Repeat(dim=N, child=mask_M)
    normalized = normalize(repeated)

    assert len(normalized.num.factors) == 1
    factor = next(iter(normalized.num.factors))
    assert isinstance(factor, NormalizedTensor)
    assert factor.dims == frozenset({M, N})
    assert isinstance(factor.tag, NormalizedTagCond)
    # Leaves are still scalar 1 and 0 — no Repeat wrapping needed for scalars.
    assert factor.tag.if_true == NormalizedExpr.of(NormalizedProduct(const=1.0))
    assert factor.tag.if_false == NormalizedExpr.of(NormalizedProduct(const=0.0))


def test_reduce_of_t_times_repeated_mask_folds(reset):
    """
    End-to-end: stile-idiomatic `T_{M,N} * Repeat(N, mask_M)` inside a
    `sum over N` folds the mask into the reduce's domain via the full
    chain — Repeat push-through exposes the tag, `mul` push-through
    distributes into it, and the reduce extractor absorbs the identity-
    else branch into the reduce's domain.
    """
    M = dim("RTF_M", 32)
    N = dim("RTF_N", 64)
    i = LoopVariable("RTF_M")

    mask_M = mask_expr((M,), domain([i], [ge(i, 8)]))
    mask_MN = Repeat(dim=N, child=mask_M)
    T = Tensor(dims=(M, N))

    reduce_expr = Reduce(op="sum", dim=N, child=BinaryOp(op="*", lhs=T, rhs=mask_MN))
    normalized = normalize(reduce_expr)

    assert len(normalized.num.factors) == 1
    factor = next(iter(normalized.num.factors))
    assert isinstance(factor, NormalizedReduce)
    assert factor.dim == N
    # Domain carries both the N-range and the M-guard from the mask.
    assert len(factor.domain.disjuncts) == 1
    conj = next(iter(factor.domain.disjuncts))
    assert len(conj) == 3


def test_tensors_with_different_tags_compare_unequal(reset):
    """Distinct tags make tensors distinct; the normalizer doesn't over-merge."""
    N = dim("TT_N_diff", 64)
    v = LoopVariable("TT_N_diff")
    tagged_a = normalize(Tensor(
        dims=(N,),
        tag=TagCond(
            stile.range_domain(v, 0, 32),
            Constant(1.0),
            Constant(0.0),
        ),
    ))
    tagged_b = normalize(Tensor(
        dims=(N,),
        tag=TagCond(
            stile.range_domain(v, 0, 48),
            Constant(1.0),
            Constant(0.0),
        ),
    ))
    assert tagged_a != tagged_b


def test_max_propagates_through_mask(reset):
    """
    Constant propagation through `max` and a `Cond` mask:
      - `max(Cond(P, 0, -inf), 0) ≡ 0`: max with 0 pulls into both
        branches, collapses to `Cond(P, 0, 0)`, then to bare `0`.
      - `max(Cond(P, 1, 0), 0) ≡ Cond(P, 1, 0)`: max(1,0)=1 and
        max(0,0)=0 in the two branches, so the result is the mask
        itself.
    """
    from stile.indexing import domain, lt, LoopVariable
    from stile.verification import verify_exprs_equivalent

    N = dim("MaxMaskN", 8)
    v = LoopVariable("MaxMaskN")
    P = domain([v], [lt(v, 4)])

    bias = Tensor(
        dims=(N,),
        tag=TagCond(P, Constant(0.0), Constant(float("-inf"))),
        name="_mask",
    )
    assert verify_exprs_equivalent(
        BinaryOp(op="max", lhs=bias, rhs=Constant(0.0)),
        Constant(0.0),
    )

    mult = Tensor(
        dims=(N,),
        tag=TagCond(P, Constant(1.0), Constant(0.0)),
        name="_mask",
    )
    assert verify_exprs_equivalent(
        BinaryOp(op="max", lhs=mult, rhs=Constant(0.0)),
        mult,
    )


def test_mask_minus_mask_is_zero(reset):
    """
    `mask - mask ≡ 0` for any TagCond mask. The same-pred fast path
    in `_distribute_binop_through_tag` combines branches, the
    `if_true == if_false` collapse in `make_tag_cond` returns the
    bare leaf, and the new "collapsed-to-pure-constant" unwrap at
    the fast-path's return site turns the resulting degenerate
    `Tensor(tag=Const(0))` back into a plain `0`.
    """
    from stile.indexing import domain, lt, LoopVariable
    from stile.verification import verify_exprs_equivalent

    N = dim("MaskSubN", 8)
    v = LoopVariable("MaskSubN")
    P = domain([v], [lt(v, 4)])

    mask = Tensor(
        dims=(N,),
        tag=TagCond(P, Constant(1.0), Constant(0.0)),
        name="_mask",
    )
    assert verify_exprs_equivalent(
        BinaryOp(op="-", lhs=mask, rhs=mask),
        Constant(0.0),
    )


def test_max_of_two_masks_collapses(reset):
    """
    `max(mask, mask) ≡ mask`. Without routing `binary_op("max", ...)`
    through `_distribute_binop_through_tag`, this used to nest as
    `Cond(P, 1, Cond(P, 1, 0))` — the inner same-P Cond stayed wrapped
    in a NormalizedExpr instead of being exposed at the outer's
    branch, so the `make_tag_cond` same-pred collapse never fired.
    """
    from stile.indexing import domain, lt, LoopVariable
    from stile.verification import verify_exprs_equivalent

    N = dim("MaxTwoMasksN", 8)
    v = LoopVariable("MaxTwoMasksN")
    P = domain([v], [lt(v, 4)])
    m = Tensor(
        dims=(N,),
        tag=TagCond(P, Constant(1.0), Constant(0.0)),
        name="_mask",
    )
    assert verify_exprs_equivalent(BinaryOp(op="max", lhs=m, rhs=m), m)


def test_varies_with_dim_consults_tag_domain(reset):
    """
    `varies_with_dim` for a `NormalizedTensor` must also consider
    the tag's free dim variables, not just `dims`. Regression: a
    tagged tensor synthesized with `dims=frozenset()` (from
    `_leaf_to_expr` wrapping a `Cond(P, …)`) was reported as
    dim-invariant, which could lead `_hoist_invariants_from_repeat`
    or `_pull_common_outer_reduce` to incorrectly move that tensor
    out of a Reduce / Repeat over `P`'s dim.
    """
    from stile.indexing import domain, lt, LoopVariable
    from stile.verification import (
        normalize, varies_with_dim, NormalizedTensor, NormalizedTagCond,
        NormalizedExpr, NormalizedProduct,
    )

    N = dim("VWDN", 8)
    v = LoopVariable("VWDN")
    P = domain([v], [lt(v, 4)])
    tag = NormalizedTagCond(
        domain=P,
        if_true=NormalizedExpr.of(NormalizedProduct(const=1.0)),
        if_false=NormalizedExpr.of(NormalizedProduct(const=0.0)),
    )
    # Crucially: dims=frozenset() but tag references N.
    synthetic = NormalizedTensor(dims=frozenset(), tag=tag, name="_mask")
    assert varies_with_dim(synthetic, N)


def test_mask_squared_idempotent(reset):
    """
    Boolean idempotence under multiplication: `mask * mask = mask`
    when `mask = Cond(P, 1, 0)`. Naively, `_distribute_binop_through_tag`
    nests the second mask inside the first's tag tree; the same-predicate
    fast path in that function collapses the nesting branch-by-branch
    into a single `Cond(P, 1*1, 0*0) = Cond(P, 1, 0)`.
    """
    from stile.specification import parse_spec_into_type
    from stile.verification import verify_exprs_equivalent
    N = dim("IdemN", 8)
    M = dim("IdemM", 8)

    mask_once = parse_spec_into_type("X:IdemN IdemM where IdemN <= IdemM")
    # `(X*mask) * (X*mask) / X = X * mask^2`. With idempotence, equals
    # `X * mask = lhs`.
    mask_sq = parse_spec_into_type(
        "(X:IdemN IdemM where IdemN <= IdemM) "
        "* (X:IdemN IdemM where IdemN <= IdemM) "
        "/ X:IdemN IdemM"
    )
    assert verify_exprs_equivalent(mask_once.et, mask_sq.et)
