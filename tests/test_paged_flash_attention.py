"""
Paged flash attention as a rolled `fori_loop` with declared invariants
on `(m, l, o)`, verifying against a gather-form softmax-attention
spec. KV cache lives in a physical pool; a `page_table` indirects each
logical position to its physical row. The kernel reads via per-tile
gathers; the spec references the same `gather`s structurally, so the
verifier matches them without inspecting `page_table`'s contents.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax
from stile import dim, reset_stile, RuntimeScalar
from stile.specification import parse_spec_into_type
from stile.verification import verify_exprs_equivalent


@pytest.fixture
def reset():
    yield
    reset_stile()


def _softmax_attention_jnp(q, k, v, *, causal : bool):
    """
    Reference softmax attention used by the paged-attention numerical
    checks. `k`/`v` are the *logical* K/V tensors — for the paged
    kernel they're produced by `K_pool[page_table]` /
    `V_pool[page_table]`. `causal=True` masks each query to keys at
    positions `[0, q_idx]`.
    """
    qctx_size = q.shape[0]
    nctx_size = k.shape[0]
    dhead = q.shape[-1]
    qk = jnp.einsum("qd,nd->qn", q, k)
    if causal:
        q_idx = jnp.arange(qctx_size)[:, None]
        k_idx = jnp.arange(nctx_size)[None, :]
        qk = jnp.where(k_idx <= q_idx, qk, -jnp.inf)
    qk = qk - jnp.max(qk, axis=-1, keepdims=True)
    logits = jnp.exp(qk)
    softmax = logits / jnp.sum(logits, axis=-1, keepdims=True)
    return jnp.einsum("qn,nd->qd", softmax, v)


def test_causal_paged_flash_attention_via_invariant(reset):
    """
    Causal paged flash attention, wired via `@tjax.jit`. KV cache pool
    of shape `(N_phys, dhead)`; `page_table` maps each logical
    position to a physical row. Per-tile bias-form causal mask
    `N_log <= qctx` layered on top of the gather indirection. The
    decorator discharges spec equivalence on the first call (running
    the function with symbolic carries through `fori_loop`'s invariant
    machinery), then `jax.jit`-compiles a wrapper for the numerical
    pass. Numerically compared against a reference causal softmax
    attention on the logical K/V (`K_pool[page_table.arr]`).
    """
    dhead = dim("dhead", 4)
    qctx = dim("qctx", 8)
    N_phys = dim("N_phys", 32)
    N_log = dim("N_log", 8)
    BN = 4

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = tjax.random.normal(k1, qctx, dhead, name="Q")
    K_pool = tjax.random.normal(k2, N_phys, dhead, name="K_pool")
    V_pool = tjax.random.normal(k3, N_phys, dhead, name="V_pool")
    page_table = tjax.runtime_index("page_table", N_log, values_in=N_phys)

    qk = (
        "(Q:qctx dhead, "
        "gather[N_phys](K_pool:N_phys dhead, page_table:N_log) "
        "-> qctx N_log)"
    )
    v_log = "gather[N_phys](V_pool:N_phys dhead, page_table:N_log)"
    pred = f"N_log < {BN} * k && N_log <= qctx"
    m_inv = f"max[N_log where {pred}]{qk}"
    l_inv = f"sum[N_log where {pred}](exp({qk} - {m_inv}))"
    o_inv = f"sum[N_log where {pred}](exp({qk} - {m_inv}) * {v_log})"

    @tjax.jit(
        spec=f"(softmax[N_log where N_log <= qctx]({qk}), "
             f"{v_log} -> qctx dhead) -> "
    )
    def paged_causal_attn(Q, K_pool, V_pool, page_table):
        def body(k, state):
            m, l, o = state
            page_tile = page_table.slice(N_log, k * BN, (k + 1) * BN)
            k_tile = K_pool.gather(N_phys, page_tile)
            v_tile = V_pool.gather(N_phys, page_tile)
            qk_tile = tjax.einsum(
                Q, k_tile, "qctx dhead, N_log dhead -> N_log qctx",
            )
            qk_tile = qk_tile + tjax.mask(
                qk_tile.type.st, "N_log <= qctx", 0.0, -jnp.inf,
            )
            tile_max = qk_tile.max(N_log)
            new_max = tjax.maximum(m, tile_max)
            logits = tjax.exp(qk_tile - new_max.repeat(qk_tile.type.st[0]))
            tile_l = logits.sum(N_log)
            new_l = tjax.exp(m - new_max) * l + tile_l
            v_proj = tjax.einsum(
                logits, v_tile, "N_log qctx, N_log dhead -> qctx dhead",
            )
            new_o = (
                tjax.exp(m - new_max).repeat(dhead).rearrange(qctx, dhead) * o
                + v_proj
            )
            return (new_max, new_l, new_o)

        init_o = tjax.zeros((qctx, dhead))
        _, l_final, o_final = tjax.fori_loop(
            0, N_log.size // BN, body,
            init_val=(-jnp.inf, 0.0, init_o),
            invariant=(m_inv, l_inv, o_inv),
        )
        return o_final / l_final.repeat(dhead).rearrange(qctx, dhead)

    result = paged_causal_attn(
        Q=Q, K_pool=K_pool, V_pool=V_pool, page_table=page_table,
    )

    K_log = K_pool.arr[page_table.arr]
    V_log = V_pool.arr[page_table.arr]
    expected = _softmax_attention_jnp(Q.arr, K_log, V_log, causal=True)
    assert jnp.allclose(result.arr, expected, atol=1e-5)


def test_paged_flash_attention_dynamic_n_used_pages(reset):
    """
    Decode-style paged attention with a runtime-known number of used
    pages: the loop's upper bound is a `RuntimeScalar`, not a
    constant. Wired via `@tjax.jit(spec=...)` — the decorator runs
    the function once with `n_used_pages` as a bare `SymbolicInt` to
    discharge the invariant + spec match (verifier never needs the
    concrete value, which is the whole point of early-exit decode),
    then `jax.jit`-compiles a wrapper that re-binds the kwarg as a
    tracer for actual execution. The body's `tjax.fori_loop` lowers
    to `jax.lax.fori_loop` once the upper bound is a tracer.
    Numerically compared to a reference attention restricted to the
    first `n_used_pages * BN` logical positions.
    """
    dhead = dim("dhead", 4)
    qctx = dim("qctx", 1)
    N_phys = dim("N_phys", 32)
    N_log = dim("N_log", 16)
    BN = 4

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = tjax.random.normal(k1, qctx, dhead, name="Q")
    K_pool = tjax.random.normal(k2, N_phys, dhead, name="K_pool")
    V_pool = tjax.random.normal(k3, N_phys, dhead, name="V_pool")
    page_table = tjax.runtime_index("page_table", N_log, values_in=N_phys)
    # `n_used_pages * BN` must stay ≤ N_log.size so the kernel's
    # `[0, BN * n_used_pages)` interval is subsumed by the spec's
    # natural `[0, N_log.size)` bound. Declaring the max here lets the
    # verifier's natural-range subsumption prove that.
    tjax.runtime_scalar("n_used_pages", max_value=N_log.size // BN + 1)

    qk = (
        "(Q:qctx dhead, "
        "gather[N_phys](K_pool:N_phys dhead, page_table:N_log) "
        "-> qctx N_log)"
    )
    v_log = "gather[N_phys](V_pool:N_phys dhead, page_table:N_log)"
    pred = f"N_log < {BN} * k"
    m_inv = f"max[N_log where {pred}]{qk}"
    l_inv = f"sum[N_log where {pred}](exp({qk} - {m_inv}))"
    o_inv = f"sum[N_log where {pred}](exp({qk} - {m_inv}) * {v_log})"

    @tjax.jit(
        spec=f"(softmax[N_log where N_log < {BN} * n_used_pages]({qk}), "
             f"{v_log} -> qctx dhead) -> "
    )
    def decode(Q, K_pool, V_pool, page_table, n_used_pages):
        def body(k, state):
            m, l, o = state
            page_tile = page_table.slice(N_log, k * BN, (k + 1) * BN)
            k_tile = K_pool.gather(N_phys, page_tile)
            v_tile = V_pool.gather(N_phys, page_tile)
            qk_tile = tjax.einsum(
                Q, k_tile, "qctx dhead, N_log dhead -> N_log qctx",
            )
            tile_max = qk_tile.max(N_log)
            new_max = tjax.maximum(m, tile_max)
            logits = tjax.exp(qk_tile - new_max.repeat(qk_tile.type.st[0]))
            tile_l = logits.sum(N_log)
            new_l = tjax.exp(m - new_max) * l + tile_l
            v_proj = tjax.einsum(
                logits, v_tile, "N_log qctx, N_log dhead -> qctx dhead",
            )
            new_o = (
                tjax.exp(m - new_max).repeat(dhead).rearrange(qctx, dhead) * o
                + v_proj
            )
            return (new_max, new_l, new_o)

        init_o = tjax.zeros((qctx, dhead))
        _, l_final, o_final = tjax.fori_loop(
            0, n_used_pages, body,
            init_val=(-jnp.inf, 0.0, init_o),
            invariant=(m_inv, l_inv, o_inv),
        )
        return o_final / l_final.repeat(dhead).rearrange(qctx, dhead)

    # Run with a concrete `n_used_pages = 3`. Verification fires on the
    # first call; subsequent calls would hit the cached jit-compiled
    # version.
    n_used = 3
    result = decode(
        Q=Q, K_pool=K_pool, V_pool=V_pool,
        page_table=page_table, n_used_pages=n_used,
    )

    # Reference: standard softmax attention over the first n_used*BN
    # logical positions (the rest of the KV cache is unused).
    K_log = K_pool.arr[page_table.arr]
    V_log = V_pool.arr[page_table.arr]
    used = n_used * BN
    qk_full = jnp.einsum("qd,nd->qn", Q.arr, K_log[:used])
    qk_full = qk_full - jnp.max(qk_full, axis=-1, keepdims=True)
    weights = jnp.exp(qk_full)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    expected = jnp.einsum("qn,nd->qd", weights, V_log[:used])
    assert jnp.allclose(result.arr, expected, atol=1e-5)


def test_paged_flash_attention_via_invariant(reset):
    """
    Non-causal paged flash attention via `@tjax.jit`. Physical KV pool
    of size `N_phys`, logical sequence of size `N_log` reachable via
    `page_table`. Kernel walks `N_log` in tiles of `BN`, slicing the
    gathered K/V per-iteration; the verifier collapses
    `K_pool.gather(page_table).slice(N_log, lo, hi)` and
    `K_pool.gather(page_table.slice(N_log, lo, hi))` to the same
    canonical Reduce-over-Gather form, then tile-merges across
    iterations. Numerically compared against a reference softmax
    attention on the logical K/V.
    """
    dhead = dim("dhead", 4)
    qctx = dim("qctx", 8)
    N_phys = dim("N_phys", 32)
    N_log = dim("N_log", 8)
    BN = 4

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = tjax.random.normal(k1, qctx, dhead, name="Q")
    K_pool = tjax.random.normal(k2, N_phys, dhead, name="K_pool")
    V_pool = tjax.random.normal(k3, N_phys, dhead, name="V_pool")
    page_table = tjax.runtime_index("page_table", N_log, values_in=N_phys)

    qk = (
        "(Q:qctx dhead, "
        "gather[N_phys](K_pool:N_phys dhead, page_table:N_log) "
        "-> qctx N_log)"
    )
    v_log = "gather[N_phys](V_pool:N_phys dhead, page_table:N_log)"
    pred = f"N_log < {BN} * k"
    m_inv = f"max[N_log where {pred}]{qk}"
    l_inv = f"sum[N_log where {pred}](exp({qk} - {m_inv}))"
    o_inv = f"sum[N_log where {pred}](exp({qk} - {m_inv}) * {v_log})"

    @tjax.jit(spec=f"(softmax[N_log]({qk}), {v_log} -> qctx dhead) -> ")
    def paged_attn(Q, K_pool, V_pool, page_table):
        def body(k, state):
            m, l, o = state
            page_tile = page_table.slice(N_log, k * BN, (k + 1) * BN)
            k_tile = K_pool.gather(N_phys, page_tile)
            v_tile = V_pool.gather(N_phys, page_tile)
            qk_tile = tjax.einsum(
                Q, k_tile, "qctx dhead, N_log dhead -> N_log qctx",
            )
            tile_max = qk_tile.max(N_log)
            new_max = tjax.maximum(m, tile_max)
            logits = tjax.exp(qk_tile - new_max.repeat(qk_tile.type.st[0]))
            tile_l = logits.sum(N_log)
            new_l = tjax.exp(m - new_max) * l + tile_l
            v_proj = tjax.einsum(
                logits, v_tile, "N_log qctx, N_log dhead -> qctx dhead",
            )
            new_o = (
                tjax.exp(m - new_max).repeat(dhead).rearrange(qctx, dhead) * o
                + v_proj
            )
            return (new_max, new_l, new_o)

        init_o = tjax.zeros((qctx, dhead))
        _, l_final, o_final = tjax.fori_loop(
            0, N_log.size // BN, body,
            init_val=(-jnp.inf, 0.0, init_o),
            invariant=(m_inv, l_inv, o_inv),
        )
        return o_final / l_final.repeat(dhead).rearrange(qctx, dhead)

    result = paged_attn(
        Q=Q, K_pool=K_pool, V_pool=V_pool, page_table=page_table,
    )

    K_log = K_pool.arr[page_table.arr]
    V_log = V_pool.arr[page_table.arr]
    expected = _softmax_attention_jnp(Q.arr, K_log, V_log, causal=False)
    assert jnp.allclose(result.arr, expected, atol=1e-5)
