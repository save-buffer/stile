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
    Outer qctx loop rolled via `tjax.fori_loop`; inner nctx loop unrolled.
    Each qctx tile is independent (no accumulator across iterations), so
    this roll is fully verifiable: the body's output type matches the spec
    parametrically, and `done()` checks that the symbolic `iq*32`/`iq*32+32`
    bounds tile the full qctx dimension as `iq` ranges over `[0, 4)`.
    """
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    dhead, qctx, nctx = dim('dhead', 16), dim('qctx', 128), dim('nctx', 512)

    Q = tjax.random.normal(k1, qctx, dhead)
    K = tjax.random.normal(k2, nctx, dhead)
    V = tjax.random.normal(k3, nctx, dhead)

    L = tjax.TypedResult(
        "(softmax[nctx]((qctx dhead, nctx dhead -> qctx nctx) / sqrt(16)), "
        "nctx dhead -> qctx dhead)"
    )

    qctx_tile_size = 32
    nctx_tile_size = 32
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


def test_flash_attention_rolled_inner(reset):
    """
    Inner nctx loop rolled via `tjax.fori_loop` — the paged-attention shape.
    `fori_loop` folds the body over `range(n_tiles)` during verification,
    producing the full accumulated carry; the normalizer collapses the
    unfolded expression and verifies the final `o` against the full
    attention spec.
    """
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    dhead, qctx, nctx = dim('dhead', 16), dim('qctx', 128), dim('nctx', 512)

    Q = tjax.random.normal(k1, qctx, dhead)
    K = tjax.random.normal(k2, nctx, dhead)
    V = tjax.random.normal(k3, nctx, dhead)

    L = tjax.TypedResult(
        "(softmax[nctx]((qctx dhead, nctx dhead -> qctx nctx) / sqrt(16)), "
        "nctx dhead -> qctx dhead)"
    )

    qctx_tile_size = 32
    nctx_tile_size = 32
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
