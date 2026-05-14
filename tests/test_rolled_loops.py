"""
Tests for rolled loops — loops whose body is verified once parametrically
instead of once per concrete iteration. Each typed backend uses its native
loop idiom: `tjax.fori_loop` mirrors `jax.lax.fori_loop`; numpy/torch will
use `with stile.loop(...)` (TODO).
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax

from stile import dim, reset_stile
from stile.type import dim_name


@pytest.fixture
def reset():
    yield
    reset_stile()


def _causal_flash_attention_jnp(q, k, v):
    """
    Causal softmax attention reference for the decode-shaped layout
    where the `qctx` queries are the *last* `qctx_size` positions of a
    longer `nctx`-length sequence. Query `q` attends to keys at
    absolute positions `[0, nctx_size - qctx_size + q]` — i.e. the
    causal mask is `k <= nctx_size - qctx_size + q`. With
    `qctx_size == nctx_size` this reduces to the usual `k <= q`.
    """
    qctx_size = q.shape[0]
    nctx_size = k.shape[0]
    dhead = q.shape[-1]
    offset = nctx_size - qctx_size
    qk = jnp.einsum("qd,nd->qn", q, k) / jnp.sqrt(dhead)
    q_idx = jnp.arange(qctx_size)[:, None]
    k_idx = jnp.arange(nctx_size)[None, :]
    qk = jnp.where(k_idx <= q_idx + offset, qk, -jnp.inf)
    qk = qk - jnp.max(qk, axis=-1, keepdims=True)
    logits = jnp.exp(qk)
    softmax = logits / jnp.sum(logits, axis=-1, keepdims=True)
    return jnp.einsum("qn,nd->qd", softmax, v)


def test_rolled_tiling_covers_output(reset):
    """
    A rolled loop that tiles the M dimension in strides of 8 over [0, 32)
    covers the full output — `done()` should accept.
    """
    M, N = dim('M', 32), dim('N', 16)
    X = tjax.random.normal(jax.random.PRNGKey(0), M, N)
    L = tjax.TypedResult("M N")
    tile = 8

    def body(i, _):
        start = i * tile
        L.assign(X.slice(M, start, start + tile))
        return None

    tjax.fori_loop(0, M.size // tile, body, None)
    L.done()


def test_rolled_tiling_gap_is_caught(reset):
    """
    Only 3 tiles of size 8 assigned out of 4 needed — `done()` should refuse.
    """
    M, N = dim('M', 32), dim('N', 16)
    X = tjax.random.normal(jax.random.PRNGKey(0), M, N)
    L = tjax.TypedResult("M N")
    tile = 8

    def body(i, _):
        start = i * tile
        L.assign(X.slice(M, start, start + tile))
        return None

    tjax.fori_loop(0, 3, body, None)
    with pytest.raises(ValueError, match="only covered up to"):
        L.done()


def test_rolled_tiling_overlap_is_caught(reset):
    """
    Tiles of size 8 but stride 4 — overlapping — `done()` should refuse.
    """
    M, N = dim('M', 32), dim('N', 16)
    X = tjax.random.normal(jax.random.PRNGKey(0), M, N)
    L = tjax.TypedResult("M N")
    tile = 8

    def body(i, _):
        start = i * 4
        L.assign(X.slice(M, start, start + tile))
        return None

    tjax.fori_loop(0, M.size // 4, body, None)
    with pytest.raises(ValueError, match="gap or overlap"):
        L.done()


def test_flash_attention_rolled_outer(reset):
    """
    Decode-shaped causal flash attention with the outer qctx loop
    rolled via `tjax.fori_loop` and the inner nctx loop unrolled.
    Each qctx tile is independent (no accumulator across iterations),
    so the roll is fully verifiable: the body's output type matches
    the per-tile causal spec parametrically, and `done()` checks the
    symbolic bounds tile the full qctx dimension. Shapes are
    rectangular (`nctx >> qctx`) to mirror the realistic decode case
    where a handful of new query positions attend to a long KV cache;
    the causal mask uses `nctx <= qctx + offset` with `offset =
    nctx - qctx` so the qctx positions sit at the end of the
    sequence. Numerically verified against a reference causal
    attention.
    """
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    dhead, qctx, nctx = dim("dhead", 16), dim("qctx", 16), dim("nctx", 128)

    Q = tjax.random.normal(k1, qctx, dhead)
    K = tjax.random.normal(k2, nctx, dhead)
    V = tjax.random.normal(k3, nctx, dhead)

    # qctx queries sit at absolute positions `[nctx - qctx, nctx)`.
    # Causal mask: query at local position `q` sees keys at `[0, q + offset]`.
    offset = nctx.size - qctx.size

    L = tjax.TypedResult(
        f"(softmax[nctx where nctx <= qctx + {offset}]"
        f"((qctx dhead, nctx dhead -> qctx nctx) / sqrt({dhead.size})), "
        f"nctx dhead -> qctx dhead)"
    )

    qctx_tile_size = 8
    nctx_tile_size = 16
    n_qctx_tiles = qctx.size // qctx_tile_size

    def body(iq, _carry):
        iqctx = iq * qctx_tile_size

        running_max = -jnp.inf
        running_l = 0
        o = 0

        for ictx in range(0, nctx.size, nctx_tile_size):
            q_tile = Q.slice(qctx, iqctx, iqctx + qctx_tile_size)
            k_tile = K.slice(nctx, ictx, ictx + nctx_tile_size)
            qk_tile = tjax.einsum(
                q_tile, k_tile, "qctx dhead, nctx dhead -> nctx qctx",
            ) / jnp.sqrt(dhead.size)
            qk_tile = qk_tile + tjax.mask(
                qk_tile.type.st, f"nctx <= qctx + {offset}", 0.0, -jnp.inf,
            )
            tile_max = qk_tile.max(nctx)
            logits = tjax.exp(qk_tile - tile_max.repeat(qk_tile.type.st[0]))

            tile_l = logits.sum(nctx)
            new_max = tjax.maximum(tile_max, running_max)
            new_l = (
                tjax.exp(running_max - new_max) * running_l
                + tjax.exp(tile_max - new_max) * tile_l
            )

            v_tile = V.slice(nctx, ictx, ictx + nctx_tile_size)
            v_proj = tjax.einsum(
                logits, v_tile, "nctx qctx, nctx dhead -> qctx dhead",
            )

            rescaled_old_o = (
                (running_l * tjax.exp(running_max - new_max))
                .repeat(dhead).rearrange(qctx, dhead) * o
            )
            rescaled_v_proj = (
                tjax.exp(tile_max - new_max)
                .repeat(dhead).rearrange(qctx, dhead) * v_proj
            )

            o = (rescaled_old_o + rescaled_v_proj) / (
                new_l.repeat(dhead).rearrange(qctx, dhead)
            )
            running_l = new_l
            running_max = new_max

        L.assign(o)
        return _carry

    tjax.fori_loop(0, n_qctx_tiles, body, None)
    L.done()

    expected = _causal_flash_attention_jnp(Q.arr, K.arr, V.arr)
    assert jnp.allclose(L.arr, expected, atol=1e-5)


def test_flash_attention_rolled_inner(reset):
    """
    Decode-shaped causal flash attention with the inner nctx loop
    rolled via `tjax.fori_loop` (the paged-attention shape) and the
    outer qctx loop unrolled. `fori_loop` folds the body over the
    iteration range during verification, producing the full
    accumulated carry; the normalizer collapses the unfolded
    expression and verifies the final `o` against the per-tile causal
    spec. Same rectangular shape and offset as the outer-rolled test.
    Numerically verified against a reference causal attention.
    """
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    dhead, qctx, nctx = dim("dhead", 16), dim("qctx", 16), dim("nctx", 128)

    Q = tjax.random.normal(k1, qctx, dhead)
    K = tjax.random.normal(k2, nctx, dhead)
    V = tjax.random.normal(k3, nctx, dhead)

    offset = nctx.size - qctx.size

    L = tjax.TypedResult(
        f"(softmax[nctx where nctx <= qctx + {offset}]"
        f"((qctx dhead, nctx dhead -> qctx nctx) / sqrt({dhead.size})), "
        f"nctx dhead -> qctx dhead)"
    )

    qctx_tile_size = 8
    nctx_tile_size = 16
    n_nctx_tiles = nctx.size // nctx_tile_size

    for iqctx in range(0, qctx.size, qctx_tile_size):
        q_tile = Q.slice(qctx, iqctx, iqctx + qctx_tile_size)

        def body(ictx_idx, carry):
            running_max, running_l, o = carry
            ictx = ictx_idx * nctx_tile_size

            k_tile = K.slice(nctx, ictx, ictx + nctx_tile_size)
            qk_tile = tjax.einsum(
                q_tile, k_tile, "qctx dhead, nctx dhead -> nctx qctx",
            ) / jnp.sqrt(dhead.size)
            qk_tile = qk_tile + tjax.mask(
                qk_tile.type.st, f"nctx <= qctx + {offset}", 0.0, -jnp.inf,
            )
            tile_max = qk_tile.max(nctx)
            logits = tjax.exp(qk_tile - tile_max.repeat(qk_tile.type.st[0]))

            tile_l = logits.sum(nctx)
            new_max = tjax.maximum(tile_max, running_max)
            new_l = (
                tjax.exp(running_max - new_max) * running_l
                + tjax.exp(tile_max - new_max) * tile_l
            )

            v_tile = V.slice(nctx, ictx, ictx + nctx_tile_size)
            v_proj = tjax.einsum(
                logits, v_tile, "nctx qctx, nctx dhead -> qctx dhead",
            )

            rescaled_old_o = (
                (running_l * tjax.exp(running_max - new_max))
                .repeat(dhead).rearrange(qctx, dhead) * o
            )
            rescaled_v_proj = (
                tjax.exp(tile_max - new_max)
                .repeat(dhead).rearrange(qctx, dhead) * v_proj
            )

            new_o = (rescaled_old_o + rescaled_v_proj) / (
                new_l.repeat(dhead).rearrange(qctx, dhead)
            )
            return (new_max, new_l, new_o)

        init_state = (-jnp.inf, 0, 0)
        _, _, o = tjax.fori_loop(0, n_nctx_tiles, body, init_state)
        L.assign(o)

    L.done()

    expected = _causal_flash_attention_jnp(Q.arr, K.arr, V.arr)
    assert jnp.allclose(L.arr, expected, atol=1e-5)
