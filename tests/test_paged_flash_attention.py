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


def test_causal_paged_flash_attention_via_invariant(reset):
    """Causal paged flash attention. KV cache pool of shape
    `(N_phys, dhead)`; `page_table` maps each logical position to a
    physical row. Per-tile bias-form causal mask `N_log <= qctx`
    layered on top of the gather indirection. The verifier discharges
    the same `(m, l, o)` invariants as the regular causal flash test
    — the gather is opaque, so as long as kernel and spec reference
    the same `page_table` tensor by name, the structural equality
    holds across the loop."""
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

    def body(k, state):
        m, l, o = state
        # Per-iter gather: read `BN` page-table entries and use them to
        # gather K/V tiles from the pool. This is what a real paged
        # kernel does — never materialize the full logical K/V tensor.
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

    init_o = tjax.zeros((qctx, dhead))
    m_final, l_final, o_final = tjax.fori_loop(
        0, N_log.size // BN, body,
        init_val=(-jnp.inf, 0.0, init_o),
        invariant=(m_inv, l_inv, o_inv),
    )

    final_attn = o_final / l_final.repeat(dhead).rearrange(qctx, dhead)
    expected = parse_spec_into_type(
        f"(softmax[N_log where N_log <= qctx]({qk}), {v_log} -> qctx dhead) -> "
    )
    assert verify_exprs_equivalent(final_attn.type.et, expected.et)


def test_paged_flash_attention_dynamic_n_used_pages(reset):
    """Decode-style paged attention with a runtime-known number of
    used pages: the loop's upper bound is a free `LoopVariable`
    (`n_used_pages`), not a constant. `_fori_loop_with_invariant`
    auto-extracts free vars from the loop bounds and adds them to the
    spec parser's `loop_vars`, so `n_used_pages` parses as a
    symbolic identifier in the invariant and the final-comparison
    spec. Kernel and spec both reference `n_used_pages` by name —
    structural equality holds without the verifier ever needing its
    concrete value, which is the whole point of early-exit decode."""
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
    n_used_pages = RuntimeScalar("n_used_pages", max_value=N_log.size // BN + 1)

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

    init_o = tjax.zeros((qctx, dhead))
    m_final, l_final, o_final = tjax.fori_loop(
        0, n_used_pages, body,
        init_val=(-jnp.inf, 0.0, init_o),
        invariant=(m_inv, l_inv, o_inv),
    )

    final_attn = o_final / l_final.repeat(dhead).rearrange(qctx, dhead)
    expected = parse_spec_into_type(
        f"(softmax[N_log where N_log < {BN} * n_used_pages]({qk}), "
        f"{v_log} -> qctx dhead) -> ",
        loop_vars={"n_used_pages"},
    )
    assert verify_exprs_equivalent(final_attn.type.et, expected.et)


def test_paged_flash_attention_via_invariant(reset):
    """Non-causal paged flash attention: physical KV pool of size
    `N_phys`, logical sequence of size `N_log` reachable via
    `page_table`. Kernel walks `N_log` in tiles of `BN`, slicing the
    gathered K/V per-iteration; the verifier collapses
    `K_pool.gather(page_table).slice(N_log, lo, hi)` and
    `K_pool.gather(page_table.slice(N_log, lo, hi))` to the same
    canonical Reduce-over-Gather form, then tile-merges across
    iterations exactly like the non-paged flash attention test."""
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

    init_o = tjax.zeros((qctx, dhead))
    m_final, l_final, o_final = tjax.fori_loop(
        0, N_log.size // BN, body,
        init_val=(-jnp.inf, 0.0, init_o),
        invariant=(m_inv, l_inv, o_inv),
    )

    final_attn = o_final / l_final.repeat(dhead).rearrange(qctx, dhead)
    expected = parse_spec_into_type(
        f"(softmax[N_log]({qk}), {v_log} -> qctx dhead) -> "
    )
    assert verify_exprs_equivalent(final_attn.type.et, expected.et)
