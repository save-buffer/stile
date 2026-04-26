"""
Tests for `NormalizedParametricReduce` and the `make_parametric_reduce`
collapse rewrites — the IR primitive for sum/max reductions induced by a
rolled loop.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax

from stile.indexing import LoopVariable, interval_domain
from stile.verification import (
    NormalizedTensor, NormalizedReduce, NormalizedParametricReduce,
    NormalizedExpr, NormalizedProduct,
    make_reduce, make_parametric_reduce, make_max, normalize,
)
from stile.type import FullDim
from stile import dim, reset_stile


@pytest.fixture
def reset():
    yield
    reset_stile()


def _leaf(d : FullDim) -> NormalizedExpr:
    return NormalizedExpr.of(NormalizedTensor(dims=frozenset({d})))


def test_empty_range_sum_is_zero(reset):
    N = FullDim("PN", 128)
    k = LoopVariable("k")
    body = NormalizedExpr.of(make_reduce(N, "sum", [(0, 32)], _leaf(N)))
    # lo == hi: sum over empty range is zero.
    result = make_parametric_reduce(k, 3, 3, "sum", body)
    assert result == NormalizedExpr.of(NormalizedProduct(const=0.0))


def test_empty_range_max_is_neg_inf(reset):
    N = FullDim("PO", 128)
    k = LoopVariable("k")
    body = NormalizedExpr.of(make_reduce(N, "max", [(0, 32)], _leaf(N)))
    result = make_parametric_reduce(k, 5, 5, "max", body)
    assert result == NormalizedExpr.of(NormalizedProduct(const=float("-inf")))


def test_loop_invariant_body_sum_scales(reset):
    N = FullDim("PP", 128)
    k = LoopVariable("k")
    # body is a plain tensor (no dependence on k): sum over 4 iterations = 4 * tensor.
    body = _leaf(N)
    result = make_parametric_reduce(k, 0, 4, "sum", body)
    expected = NormalizedExpr.of(NormalizedProduct(
        const=4.0,
        factors=body.num.factors,
    ))
    assert result == expected


def test_loop_invariant_body_max_is_identity(reset):
    N = FullDim("PQ", 128)
    k = LoopVariable("k")
    body = _leaf(N)
    result = make_parametric_reduce(k, 0, 4, "max", body)
    assert result == body


def test_tiled_sum_reduce_collapses_to_full_range(reset):
    """
    The headline rewrite: `ParametricSum(k in [0, 4), Reduce(N[k*32 : k*32+32], sum, tensor))`
    should collapse to `Reduce(N[0 : 128], sum, tensor)`.
    """
    N = FullDim("PR", 128)
    k = LoopVariable("k")
    child = _leaf(N)
    tile_reduce = NormalizedExpr.of(
        make_reduce(N, "sum", [(k * 32, k * 32 + 32)], child)
    )
    result = make_parametric_reduce(k, 0, 4, "sum", tile_reduce)

    expected = NormalizedExpr.of(
        make_reduce(N, "sum", [(0, 128)], child)
    )
    assert result == expected


def test_tiled_max_reduce_collapses_to_full_range(reset):
    N = FullDim("PS", 128)
    k = LoopVariable("k")
    child = _leaf(N)
    tile_reduce = NormalizedExpr.of(
        make_reduce(N, "max", [(k * 32, k * 32 + 32)], child)
    )
    result = make_parametric_reduce(k, 0, 4, "max", tile_reduce)

    expected = NormalizedExpr.of(
        make_reduce(N, "max", [(0, 128)], child)
    )
    assert result == expected


def test_non_tiled_stride_does_not_collapse(reset):
    """
    Tile width (8) doesn't match stride (16) — there are gaps between tiles,
    so the collapse is unsound. The parametric form is kept wrapped.
    """
    N = FullDim("PT", 128)
    k = LoopVariable("k")
    child = _leaf(N)
    # start = 16*k, end = 16*k + 8 (stride 16, width 8 -> gaps).
    tile_reduce = NormalizedExpr.of(
        make_reduce(N, "sum", [(k * 16, k * 16 + 8)], child)
    )
    result = make_parametric_reduce(k, 0, 4, "sum", tile_reduce)

    # Should NOT collapse to a single Reduce.
    single_factor = None
    if not result.den.factors and result.num.const == 1.0 and len(result.num.factors) == 1:
        factor, _ = next(iter(result.num.factors.items()))
        single_factor = factor
    assert isinstance(single_factor, NormalizedParametricReduce)


def test_collapsed_reduce_matches_unrolled_form(reset):
    """
    The collapsed form of a tiled parametric reduce must equal the
    expression you'd get from manually unrolling and merging — confirming
    that the rewrite is consistent with the `_canonicalize_intervals`
    interval-merging path.
    """
    N = FullDim("PU", 128)
    k = LoopVariable("k")
    child = _leaf(N)

    # Parametric form.
    tile_reduce = NormalizedExpr.of(
        make_reduce(N, "sum", [(k * 32, k * 32 + 32)], child)
    )
    collapsed = make_parametric_reduce(k, 0, 4, "sum", tile_reduce)

    # Unrolled form: 4 separate reduce factors, summed, with intervals merged.
    unrolled = make_reduce(
        N, "sum",
        [(0, 32), (32, 64), (64, 96), (96, 128)],
        child,
    )
    assert collapsed == NormalizedExpr.of(unrolled)


def test_symbolic_fori_loop_emits_parametric_reduce(reset):
    """
    `fori_loop(0, N_tiles_symbolic, body, 0)` with a sum-accumulator body
    should emit a `ParametricReduce` at the `ExprType` level, and after
    normalization its body is exactly the per-iteration contribution.
    """
    N = dim("PN512", 512)
    n_tiles = LoopVariable("n_tiles")
    X = tjax.random.normal(jax.random.PRNGKey(0), N)

    def body(k, s):
        tile = X.slice(N, k * 32, k * 32 + 32)
        return s + tile.sum(N)

    result = tjax.fori_loop(0, n_tiles, body, 0)

    # With a tiled sum body, the collapse rewrite folds the ParametricReduce
    # into a single NormalizedReduce over [0, n_tiles * 32).
    normalized = normalize(result.type.et)
    assert not normalized.den.factors
    assert normalized.num.const == 1.0
    assert len(normalized.num.factors) == 1
    factor, count = next(iter(normalized.num.factors.items()))
    assert count == 1
    assert isinstance(factor, NormalizedReduce)
    assert factor.op == "sum"
    assert factor.dim == N
    assert factor.domain == interval_domain(
        LoopVariable(factor.dim.name),
        [(0, n_tiles * 32)],
    )


def test_max_absorbs_boundary_term(reset):
    """
    `max(ParametricMax(k in [0, 4), f(k)), f(4))` → `ParametricMax(k in [0, 5), f(k))`.
    We use non-tile-adjacent bounds (stride 32, width 16 → gaps between
    tiles) so the parametric doesn't collapse to a single Reduce, letting
    us observe the range-extension directly.
    """
    N = FullDim("AbsN", 256)
    k = LoopVariable("k")
    child = _leaf(N)
    body = NormalizedExpr.of(make_reduce(N, "max", [(k * 32, k * 32 + 16)], child))

    param_expr = make_parametric_reduce(k, 0, 4, "max", body)
    # Sanity: this should stay wrapped as a ParametricReduce, not collapse.
    assert isinstance(
        next(iter(param_expr.num.factors)), NormalizedParametricReduce,
    )

    # f(4) = Reduce(N[128 : 144], max, child)
    sibling = NormalizedExpr.of(make_reduce(N, "max", [(128, 144)], child))

    result = make_max([param_expr, sibling])
    expected = make_parametric_reduce(k, 0, 5, "max", body)
    assert result == expected


def test_max_absorbs_lower_boundary(reset):
    """
    `max(f(-1), ParametricMax(k in [0, 4), f(k)))` → `ParametricMax(k in [-1, 4), f(k))`.
    """
    N = FullDim("AbsNlo", 256)
    k = LoopVariable("k")
    child = _leaf(N)
    body = NormalizedExpr.of(make_reduce(N, "max", [(k * 32 + 40, k * 32 + 56)], child))

    param_expr = make_parametric_reduce(k, 0, 4, "max", body)
    # f(-1) = Reduce(N[-32 + 40 : -32 + 56], max, child) = Reduce(N[8 : 24], max, child)
    sibling = NormalizedExpr.of(make_reduce(N, "max", [(8, 24)], child))

    result = make_max([param_expr, sibling])
    expected = make_parametric_reduce(k, -1, 4, "max", body)
    assert result == expected


def test_sum_absorbs_boundary_term(reset):
    """
    `ParametricSum(k in [0, 4), f(k)) + f(4)` → `ParametricSum(k in [0, 5), f(k))`.
    """
    N = FullDim("AbsNsum", 256)
    k = LoopVariable("k")
    child = _leaf(N)
    body = NormalizedExpr.of(make_reduce(N, "sum", [(k * 32, k * 32 + 16)], child))

    param_expr = make_parametric_reduce(k, 0, 4, "sum", body)
    # f(4) = Reduce(N[128 : 144], sum, child)
    sibling = NormalizedExpr.of(make_reduce(N, "sum", [(128, 144)], child))

    # add(param_expr, sibling) routes through make_sum under the hood.
    # We use the two-argument public sub/add pathway directly via the
    # internal add helper applied to NormalizedExpr values.
    from stile.verification import add as _add  # NormalizedExpr-level add
    result = _add(param_expr, sibling)

    expected = make_parametric_reduce(k, 0, 5, "sum", body)
    assert result == expected


def test_symbolic_fori_loop_emits_parametric_max(reset):
    """
    `fori_loop(0, n_tiles, body, -inf)` with a max-accumulator body
    (`body(k, s) = max(s, tile.max(N))`) traces once with symbolic `k`,
    produces a `ParametricReduce(op="max")`, and collapses end-to-end to a
    single `NormalizedReduce` over `N[0 : n_tiles*32]` via the tiled-max
    collapse rewrite. This is the running_max chain from flash attention,
    verified mechanically with no user-supplied invariant.
    """
    N = dim("PNmaxE", 512)
    n_tiles = LoopVariable("n_tiles_maxE")
    X = tjax.random.normal(jax.random.PRNGKey(0), N)

    def body(k, s):
        tile = X.slice(N, k * 32, k * 32 + 32)
        return tjax.maximum(s, tile.max(N))

    result = tjax.fori_loop(0, n_tiles, body, -jnp.inf)

    normalized = normalize(result.type.et)
    assert not normalized.den.factors
    assert normalized.num.const == 1.0
    assert len(normalized.num.factors) == 1
    factor, count = next(iter(normalized.num.factors.items()))
    assert count == 1
    assert isinstance(factor, NormalizedReduce)
    assert factor.op == "max"
    assert factor.dim == N
    assert factor.domain == interval_domain(
        LoopVariable(factor.dim.name),
        [(0, n_tiles * 32)],
    )


def test_symbolic_fori_loop_with_non_tiling_stride(reset):
    """
    If the body's stride doesn't match the tile width, the collapse
    rewrite declines to merge and we keep a NormalizedParametricReduce.
    """
    N = dim("PN1024", 1024)
    n_tiles = LoopVariable("n_tiles_b")
    X = tjax.random.normal(jax.random.PRNGKey(0), N)

    def body(k, s):
        # stride 32, tile width 16 -> gaps between tiles.
        tile = X.slice(N, k * 32, k * 32 + 16)
        return s + tile.sum(N)

    result = tjax.fori_loop(0, n_tiles, body, 0)
    normalized = normalize(result.type.et)

    # No collapse: the parametric reduce is kept as-is.
    assert len(normalized.num.factors) == 1
    factor, _ = next(iter(normalized.num.factors.items()))
    assert isinstance(factor, NormalizedParametricReduce)
