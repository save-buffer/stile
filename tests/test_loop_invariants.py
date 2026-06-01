"""
Hoare-style loop invariants on `tjax.fori_loop`.

The user declares what the loop's state *should* be at iteration `k`
as a stile spec string referencing the free LoopVariable `k`. The
verifier discharges:

  - Base case:    `init_val == invariant[k=0]`
  - Inductive:    `body(k, invariant[k]) == invariant[k+1]`

For tuple state, pass a tuple of strings matching the state shape.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax
from stile import dim, reset_stile
from stile.specification import parse_spec_into_type
from stile.verification import verify_exprs_equivalent


def test_scalar_sum_invariant(reset):
    """
    A scalar accumulator: `acc[k+1] = acc[k] + X[k]`. Invariant is
    `sum over n in [0, k) of X[n]`.
    """
    N = dim("N", 16)
    X = tjax.random.normal(jax.random.PRNGKey(0), N, name="X")

    def body(i, acc):
        return acc + X.slice(N, i, i + 1).sum(N)

    final = tjax.fori_loop(
        0, N.size, body, init_val=0.0,
        invariant="sum[N where N < k](X:N)",
    )

    expected = parse_spec_into_type("sum[N](X:N) -> ")
    assert verify_exprs_equivalent(final.type.et, expected.et)


def test_invariant_base_case_failure_rejected(reset):
    """If `init_val` doesn't match `invariant[k=0]`, the verifier rejects."""
    N = dim("N", 16)
    X = tjax.random.normal(jax.random.PRNGKey(0), N, name="X")

    def body(i, acc):
        return acc + X.slice(N, i, i + 1).sum(N)

    with pytest.raises(AssertionError, match="base case"):
        tjax.fori_loop(
            0, N.size, body, init_val=5.0,  # wrong: should be 0
            invariant="sum[N where N < k](X:N)",
        )


def test_invariant_inductive_step_failure_rejected(reset):
    """
    If `body(k, invariant[k])` doesn't equal `invariant[k+1]`, the
    verifier rejects.
    """
    N = dim("N", 16)
    X = tjax.random.normal(jax.random.PRNGKey(0), N, name="X")

    def buggy_body(i, acc):
        # Bug: multiplies X[i] by 2 — kernel computes 2*sum(X), not sum(X).
        return acc + (X.slice(N, i, i + 1) * 2).sum(N)

    with pytest.raises(AssertionError, match="inductive step"):
        tjax.fori_loop(
            0, N.size, buggy_body, init_val=0.0,
            invariant="sum[N where N < k](X:N)",
        )


def test_tiled_dot_product_invariant(reset):
    """
    Tile-walked cumulative dot product `Σ X[i] * Y[i]`. The body
    walks `N` in tiles of `BN` and accumulates one tile's
    `(X[tile] * Y[tile]).sum(N)`. The invariant uses a symbolic upper
    bound `BN * k`, exercising symbolic-interval-merge with non-unit
    coefficients.
    """
    N = dim("N", 16)
    BN = 4
    X = tjax.random.normal(jax.random.PRNGKey(0), N, name="X")
    Y = tjax.random.normal(jax.random.PRNGKey(1), N, name="Y")

    def body(k, acc):
        x_tile = X.slice(N, k * BN, (k + 1) * BN)
        y_tile = Y.slice(N, k * BN, (k + 1) * BN)
        return acc + (x_tile * y_tile).sum(N)

    final = tjax.fori_loop(
        0, N.size // BN, body, init_val=0.0,
        invariant=f"sum[N where N < {BN} * k](X:N * Y:N)",
    )

    expected = parse_spec_into_type("sum[N](X:N * Y:N) -> ")
    assert verify_exprs_equivalent(final.type.et, expected.et)


def test_online_softmax_running_max_invariant(reset):
    """
    The simplest online-softmax piece: just track `running_max =
    max over n in [0, BN*k) of qk[n]` as a rolled loop. Each iteration
    walks one nctx tile, takes its max, and folds into the running
    max. Single scalar state, no rescaling — pure max invariant.
    """
    nctx = dim("nctx", 8)
    BN = 4
    qk = tjax.random.normal(jax.random.PRNGKey(0), nctx, name="qk")

    def body(k, m):
        tile_max = qk.slice(nctx, k * BN, (k + 1) * BN).max(nctx)
        return tjax.maximum(m, tile_max)

    final = tjax.fori_loop(
        0, nctx.size // BN, body, init_val=-jnp.inf,
        invariant=f"max[nctx where nctx < {BN} * k](qk:nctx)",
    )

    expected = parse_spec_into_type("max[nctx](qk:nctx) -> ")
    assert verify_exprs_equivalent(final.type.et, expected.et)


def test_online_softmax_aggregates_invariant(reset):
    """
    Online softmax's `(running_max, running_l)` running aggregates,
    where `l[k] = sum over n in [0, BN*k) of exp(qk[n] - m[k])` —
    crucially `l[k]` depends on `m[k]`. The body rescales `l[k]` by
    `exp(m[k] - m[k+1])` to re-anchor it on the new max. The verifier
    has to discharge the algebraic identity
    `exp(m[k] - m[k+1]) * Σ exp(qk - m[k]) = Σ exp(qk - m[k+1])`,
    which falls out of `normalize_exp` distributing exp across sums
    plus fraction cancellation.
    """
    nctx = dim("nctx", 8)
    BN = 4
    qk = tjax.random.normal(jax.random.PRNGKey(0), nctx, name="qk")

    def body(k, state):
        m, l = state
        qk_tile = qk.slice(nctx, k * BN, (k + 1) * BN)
        tile_max = qk_tile.max(nctx)
        new_max = tjax.maximum(m, tile_max)
        logits = tjax.exp(qk_tile - new_max.repeat(qk_tile.type.st[0]))
        tile_l = logits.sum(nctx)
        new_l = tjax.exp(m - new_max) * l + tile_l
        return (new_max, new_l)

    m_inv = f"max[nctx where nctx < {BN} * k](qk:nctx)"
    l_inv = (
        f"sum[nctx where nctx < {BN} * k]("
        f"exp(qk:nctx - max[nctx where nctx < {BN} * k](qk:nctx))"
        f")"
    )
    m_final, l_final = tjax.fori_loop(
        0, nctx.size // BN, body, init_val=(-jnp.inf, 0.0),
        invariant=(m_inv, l_inv),
    )

    m_expected = parse_spec_into_type("max[nctx](qk:nctx) -> ")
    l_expected = parse_spec_into_type(
        "sum[nctx](exp(qk:nctx - max[nctx](qk:nctx))) -> "
    )
    assert verify_exprs_equivalent(m_final.type.et, m_expected.et)
    assert verify_exprs_equivalent(l_final.type.et, l_expected.et)


def _softmax_jnp(qk, *, axis=-1):
    qk = qk - jnp.max(qk, axis=axis, keepdims=True)
    e = jnp.exp(qk)
    return e / jnp.sum(e, axis=axis, keepdims=True)


def _causal_attention_jnp(Q, K, V):
    """
    Reference causal softmax attention (no scaling). qctx_size ==
    nctx_size for the local tests; mask is `k_idx <= q_idx`.
    """
    qctx_size = Q.shape[0]
    nctx_size = K.shape[0]
    qk = jnp.einsum("qd,nd->qn", Q, K)
    q_idx = jnp.arange(qctx_size)[:, None]
    k_idx = jnp.arange(nctx_size)[None, :]
    qk = jnp.where(k_idx <= q_idx, qk, -jnp.inf)
    return jnp.einsum("qn,nd->qd", _softmax_jnp(qk), V)


def test_tiled_attention_qkv_via_invariant(reset):
    """
    Plain (non-softmax) QKV-style einsum walked tile-by-tile in nctx
    via `@tjax.jit`-wrapped `fori_loop` with an invariant. Single-state
    invariant `Σ_n_in_[0,BN*k) (Q · K[n]) * V[n]` exercises slice-form
    reduce inside a contraction with a body of shape `(qctx, dhead)`.
    Numerically compared to a direct `Q @ K^T @ V`-style einsum.
    """
    dhead, qctx, nctx = dim("dhead", 4), dim("qctx", 2), dim("nctx", 8)
    BN = 4
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = tjax.random.normal(k1, qctx, dhead, name="Q")
    K = tjax.random.normal(k2, nctx, dhead, name="K")
    V = tjax.random.normal(k3, nctx, dhead, name="V")

    o_inv = (
        f"sum[nctx where nctx < {BN} * k]("
        f"(Q:qctx dhead, K:nctx dhead -> qctx nctx) * V:nctx dhead"
        f")"
    )

    @tjax.jit(
        spec="((Q:qctx dhead, K:nctx dhead -> qctx nctx), "
             "V:nctx dhead -> qctx dhead) -> "
    )
    def attn(Q, K, V):
        def body(k, o):
            k_tile = K.slice(nctx, k * BN, (k + 1) * BN)
            v_tile = V.slice(nctx, k * BN, (k + 1) * BN)
            qk_tile = tjax.einsum(
                Q, k_tile, "qctx dhead, nctx dhead -> qctx nctx",
            )
            return o + tjax.einsum(
                qk_tile, v_tile, "qctx nctx, nctx dhead -> qctx dhead",
            )
        return tjax.fori_loop(
            0, nctx.size // BN, body,
            init_val=tjax.zeros((qctx, dhead)),
            invariant=o_inv,
        )

    result = attn(Q=Q, K=K, V=V)
    expected = jnp.einsum(
        "qn,nd->qd", jnp.einsum("qd,nd->qn", Q.arr, K.arr), V.arr,
    )
    assert jnp.allclose(result.arr, expected, atol=1e-5)


def test_flash_attention_via_invariant(reset):
    """
    Causal flash attention as a rolled `fori_loop` with declared
    invariants on `(m, l, o)`, wired through `@tjax.jit`. The body
    applies the causal mask `nctx <= qctx` per tile; the invariant
    carries both the iteration restriction `nctx < BN*k` and the
    causal predicate. The verifier discharges base case + inductive
    step against the Hoare-style invariants; the merge across
    iterations uses loop-var-symbolic priority in
    `_split_reduce_domain` to share the causal extras key across tile
    and running aggregate. Numerically compared to a reference causal
    attention.
    """
    dhead, qctx, nctx = dim("dhead", 4), dim("qctx", 8), dim("nctx", 8)
    BN = 4
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = tjax.random.normal(k1, qctx, dhead, name="Q")
    K = tjax.random.normal(k2, nctx, dhead, name="K")
    V = tjax.random.normal(k3, nctx, dhead, name="V")

    qk = "(Q:qctx dhead, K:nctx dhead -> qctx nctx)"
    pred = f"nctx < {BN} * k && nctx <= qctx"
    m_inv = f"max[nctx where {pred}]{qk}"
    l_inv = f"sum[nctx where {pred}](exp({qk} - {m_inv}))"
    o_inv = f"sum[nctx where {pred}](exp({qk} - {m_inv}) * V:nctx dhead)"

    @tjax.jit(
        spec="(softmax[nctx where nctx <= qctx]("
             "Q:qctx dhead, K:nctx dhead -> qctx nctx"
             "), V:nctx dhead -> qctx dhead) -> "
    )
    def causal_attn(Q, K, V):
        def body(k, state):
            m, l, o = state
            k_tile = K.slice(nctx, k * BN, (k + 1) * BN)
            v_tile = V.slice(nctx, k * BN, (k + 1) * BN)
            qk_tile = tjax.einsum(
                Q, k_tile, "qctx dhead, nctx dhead -> nctx qctx",
            )
            # Bias-form causal mask: 0 inside / -inf outside, matching
            # how the spec's `max[nctx where ...]` lowers.
            qk_tile = qk_tile + tjax.mask(
                qk_tile.type.st, "nctx <= qctx", 0.0, -jnp.inf,
            )
            tile_max = qk_tile.max(nctx)
            new_max = tjax.maximum(m, tile_max)
            logits = tjax.exp(qk_tile - new_max.repeat(qk_tile.type.st[0]))
            tile_l = logits.sum(nctx)
            new_l = tjax.exp(m - new_max) * l + tile_l
            v_proj = tjax.einsum(
                logits, v_tile, "nctx qctx, nctx dhead -> qctx dhead",
            )
            new_o = (
                tjax.exp(m - new_max).repeat(dhead).rearrange(qctx, dhead) * o
                + v_proj
            )
            return (new_max, new_l, new_o)

        init_o = tjax.zeros((qctx, dhead))
        _, l_final, o_final = tjax.fori_loop(
            0, nctx.size // BN, body,
            init_val=(-jnp.inf, 0.0, init_o),
            invariant=(m_inv, l_inv, o_inv),
        )
        return o_final / l_final.repeat(dhead).rearrange(qctx, dhead)

    result = causal_attn(Q=Q, K=K, V=V)
    expected = _causal_attention_jnp(Q.arr, K.arr, V.arr)
    assert jnp.allclose(result.arr, expected, atol=1e-5)


def test_flash_attention_via_invariant_skip_tail(reset):
    """
    Causal flash attention with `nctx > qctx` and the loop stopping
    early — only walking nctx tiles up to `qctx_size/BN`, skipping the
    tail where every position is above the causal diagonal. inv[K] is
    `n in [0, BN*K) ∩ {n ≤ q}`; the spec is `n in [0, nctx_size) ∩
    {n ≤ q}`. With `q ∈ [0, qctx_size)` and `BN*K = qctx_size ≤
    nctx_size`, both `n < BN*K` and `n < nctx_size` are subsumed by
    `n ≤ q`, so the bound-subsumption pass collapses them to the same
    canonical `{n ≥ 0, n ≤ q}` form. Numerically verified against a
    causal-masked reference attention.
    """
    dhead, qctx, nctx = dim("dhead", 4), dim("qctx", 8), dim("nctx", 16)
    BN = 4
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = tjax.random.normal(k1, qctx, dhead, name="Q")
    K = tjax.random.normal(k2, nctx, dhead, name="K")
    V = tjax.random.normal(k3, nctx, dhead, name="V")

    qk = "(Q:qctx dhead, K:nctx dhead -> qctx nctx)"
    pred = f"nctx < {BN} * k && nctx <= qctx"
    m_inv = f"max[nctx where {pred}]{qk}"
    l_inv = f"sum[nctx where {pred}](exp({qk} - {m_inv}))"
    o_inv = f"sum[nctx where {pred}](exp({qk} - {m_inv}) * V:nctx dhead)"

    @tjax.jit(
        spec="(softmax[nctx where nctx <= qctx]("
             "Q:qctx dhead, K:nctx dhead -> qctx nctx"
             "), V:nctx dhead -> qctx dhead) -> "
    )
    def causal_attn_skip_tail(Q, K, V):
        def body(k, state):
            m, l, o = state
            k_tile = K.slice(nctx, k * BN, (k + 1) * BN)
            v_tile = V.slice(nctx, k * BN, (k + 1) * BN)
            qk_tile = tjax.einsum(
                Q, k_tile, "qctx dhead, nctx dhead -> nctx qctx",
            )
            qk_tile = qk_tile + tjax.mask(
                qk_tile.type.st, "nctx <= qctx", 0.0, -jnp.inf,
            )
            tile_max = qk_tile.max(nctx)
            new_max = tjax.maximum(m, tile_max)
            logits = tjax.exp(qk_tile - new_max.repeat(qk_tile.type.st[0]))
            tile_l = logits.sum(nctx)
            new_l = tjax.exp(m - new_max) * l + tile_l
            v_proj = tjax.einsum(
                logits, v_tile, "nctx qctx, nctx dhead -> qctx dhead",
            )
            new_o = (
                tjax.exp(m - new_max).repeat(dhead).rearrange(qctx, dhead) * o
                + v_proj
            )
            return (new_max, new_l, new_o)

        init_o = tjax.zeros((qctx, dhead))
        _, l_final, o_final = tjax.fori_loop(
            0, qctx.size // BN, body,
            init_val=(-jnp.inf, 0.0, init_o),
            invariant=(m_inv, l_inv, o_inv),
        )
        return o_final / l_final.repeat(dhead).rearrange(qctx, dhead)

    result = causal_attn_skip_tail(Q=Q, K=K, V=V)
    # Reference: causal-masked over all nctx, but the kernel only walks
    # the first qctx_size positions — the tail is entirely above the
    # diagonal so contributes nothing to a causal softmax.
    expected = _causal_attention_jnp(Q.arr, K.arr, V.arr)
    assert jnp.allclose(result.arr, expected, atol=1e-5)


def test_tuple_state_invariant(reset):
    """
    Tuple state: track both `sum X[n]` and `sum X[n]^2` simultaneously.
    Invariant is a tuple of two specs.

    Note: the squared accumulator uses `(slice * slice).sum(N)` rather
    than `slice.sum(N) * slice.sum(N)`. They're algebraically equal
    only because the slice has a single element; the verifier folds
    the former into a clean `Reduce(sum, [k,k+1), X*X)` that
    tile-merges with the running invariant, while the latter would
    leave a square-of-a-reduce that doesn't fold.
    """
    N = dim("N", 8)
    X = tjax.random.normal(jax.random.PRNGKey(0), N, name="X")

    def body(i, state):
        s, s2 = state
        x_slice = X.slice(N, i, i + 1)
        return (
            s + x_slice.sum(N),
            s2 + (x_slice * x_slice).sum(N),
        )

    s_final, s2_final = tjax.fori_loop(
        0, N.size, body, init_val=(0.0, 0.0),
        invariant=(
            "sum[N where N < k](X:N)",
            "sum[N where N < k](X:N * X:N)",
        ),
    )

    s_expected = parse_spec_into_type("sum[N](X:N) -> ")
    s2_expected = parse_spec_into_type("sum[N](X:N * X:N) -> ")
    assert verify_exprs_equivalent(s_final.type.et, s_expected.et)
    assert verify_exprs_equivalent(s2_final.type.et, s2_expected.et)
