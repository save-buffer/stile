"""
Tile-walking causal kernel — demonstrates skipping nctx tiles that are
entirely above the causal diagonal of the current qctx tile.

Two variants:

  - `test_causal_qk_sum_tile_walking`: a *fully verifiable* causal
    accumulator. Each visited tile applies `tjax.where("nctx <= qctx")`
    to its scores and sums them; the verifier folds every tile's mask
    into its reduce's domain and merges adjacent tile-reductions. The
    spec uses the same `where` clause and converges. This is the
    structural payoff of the tile-skip optimization.

  - `test_causal_attention_tile_walking_numerical`: full causal attention
    via online softmax. Online softmax requires the bias-mask form to
    stay numerically correct, and the normalizer doesn't yet collapse
    bias-form to multiplicative-form (task #7), so this variant only
    asserts numerical equivalence to a reference jnp implementation.
    Structural verification follows once the bias-form rule lands.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax
from stile import reset_stile, dim


@pytest.fixture
def reset():
    yield
    reset_stile()


def _softmax_jnp(x, axis=-1):
    ex = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return ex / jnp.sum(ex, axis=axis, keepdims=True)


def _causal_attention_jnp(q, k, v):
    qctx_size, dhead_size = q.shape
    nctx_size = k.shape[0]
    qk = jnp.einsum('qd,nd->qn', q, k) / jnp.sqrt(dhead_size)
    q_idx = jnp.arange(qctx_size)[:, None]
    k_idx = jnp.arange(nctx_size)[None, :]
    qk = jnp.where(k_idx <= q_idx, qk, -jnp.inf)
    logits = _softmax_jnp(qk, axis=-1)
    return jnp.einsum('qn,nd->qd', logits, v)


def test_causal_qk_sum_tile_walking(reset):
    """
    Tile-walking causal accumulator: for each qctx tile, only walk the
    nctx tiles up to (and including) the diagonal one, applying the
    `where` mask uniformly. Verifier proves the full-spec equivalence.
    """
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    dhead = dim('dhead', 16)
    qctx = dim('qctx', 16)
    nctx = dim('nctx', 16)

    Q = tjax.random.normal(k1, qctx, dhead)
    K = tjax.random.normal(k2, nctx, dhead)

    qctx_tile_size = 4
    nctx_tile_size = 4

    tiles_visited = 0
    tiles_skipped = 0
    output_per_qtile : list[jnp.ndarray] = []

    for iqctx in range(0, qctx.size, qctx_tile_size):
        accum = 0
        max_ictx_inclusive = iqctx + qctx_tile_size - 1
        for ictx in range(0, nctx.size, nctx_tile_size):
            if ictx > max_ictx_inclusive:
                tiles_skipped += 1
                continue
            tiles_visited += 1

            q_tile = Q.slice(qctx, iqctx, iqctx + qctx_tile_size)
            k_tile = K.slice(nctx, ictx, ictx + nctx_tile_size)
            qk_tile = tjax.einsum(
                q_tile, k_tile, "qctx dhead, nctx dhead -> qctx nctx",
            )
            qk_tile = qk_tile.where("nctx <= qctx")
            tile_sum = qk_tile.sum(nctx)
            accum = accum + tile_sum

        # Per-qctx-tile structural verification. Slice override on nctx
        # restricts the spec's reduction range to what the kernel walked;
        # the `where` clause and the kernel's per-tile masks combine to
        # cover the same `(q, k)` set.
        accum.assert_equivalent(
            "(sum[nctx]((qctx dhead, nctx dhead -> qctx nctx) where nctx <= qctx) -> qctx)",
            nctx[:(iqctx + qctx_tile_size)],
        )
        output_per_qtile.append(accum.arr)

    # Numerical sanity check: concatenated kernel output equals the
    # straightforward causal-masked sum.
    output = jnp.concatenate(output_per_qtile, axis=0)
    qk_full = jnp.einsum("qd,nd->qn", Q.arr, K.arr)
    q_idx = jnp.arange(qctx.size)[:, None]
    k_idx = jnp.arange(nctx.size)[None, :]
    expected = jnp.where(k_idx <= q_idx, qk_full, 0.0).sum(axis=-1)
    assert jnp.allclose(output, expected, atol=1e-5)

    n_tiles = qctx.size // qctx_tile_size
    assert tiles_visited == n_tiles * (n_tiles + 1) // 2
    assert tiles_skipped == n_tiles * (n_tiles - 1) // 2


_CAUSAL_SPEC = (
    "(softmax[nctx where nctx <= qctx]"
    "((qctx dhead, nctx dhead -> qctx nctx) / sqrt(16)), "
    "nctx dhead -> qctx dhead)"
)


def test_causal_attention_tile_walking_full(reset):
    """
    Full causal flash attention with tile-skipping AND structural
    verification. The bias mask is built via `tjax.mask(..., 0, -inf)`
    so its `TagCond` lands in the AST; #7 (bias→mult convergence under
    exp) and #10 (max-side -inf absorption / tagged max-reduce fusion)
    let the per-tile online-softmax accumulator collapse to the
    explicit causal-attention spec (with `where nctx <= qctx`).
    """
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    dhead = dim('dhead', 16)
    qctx = dim('qctx', 32)
    nctx = dim('nctx', 32)

    Q = tjax.random.normal(k1, qctx, dhead)
    K = tjax.random.normal(k2, nctx, dhead)
    V = tjax.random.normal(k3, nctx, dhead)

    qctx_tile_size = 8
    nctx_tile_size = 8

    tiles_visited = 0
    tiles_skipped = 0
    o_per_qtile : list[jnp.ndarray] = []

    for iqctx in range(0, qctx.size, qctx_tile_size):
        running_max = -jnp.inf
        running_l = 0
        o = 0

        max_ictx_inclusive = iqctx + qctx_tile_size - 1
        for ictx in range(0, nctx.size, nctx_tile_size):
            if ictx > max_ictx_inclusive:
                tiles_skipped += 1
                continue
            tiles_visited += 1

            q_tile = Q.slice(qctx, iqctx, iqctx + qctx_tile_size)
            k_tile = K.slice(nctx, ictx, ictx + nctx_tile_size)

            qk_tile = tjax.einsum(
                q_tile, k_tile, "qctx dhead, nctx dhead -> nctx qctx",
            ) / jnp.sqrt(dhead.size)

            # Causal bias: `0` where `nctx <= qctx`, `-inf` outside. The
            # `mask` intrinsic puts a `TagCond(D, 0, -inf)` into the AST
            # so the verifier can see the predicate; the runtime
            # array carries the actual `0`/`-inf` values for online
            # softmax to stay numerically correct on every tile.
            qk_tile = qk_tile + tjax.mask(
                qk_tile.type.st, "nctx <= qctx", 0.0, -jnp.inf,
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
                running_l * tjax.exp(running_max - new_max)
            ).repeat(dhead).rearrange(qctx, dhead) * o
            rescaled_v_proj = tjax.exp(
                tile_max - new_max,
            ).repeat(dhead).rearrange(qctx, dhead) * v_proj

            o = (rescaled_old_o + rescaled_v_proj) / new_l.repeat(
                dhead,
            ).rearrange(qctx, dhead)
            running_l = new_l
            running_max = new_max

        # Structural verification per qctx tile: kernel's accumulated
        # output should normalize to the explicit causal attention spec
        # restricted to nctx[:max_walked].
        o.assert_equivalent(_CAUSAL_SPEC, nctx[:(iqctx + qctx_tile_size)])
        o_per_qtile.append(o.arr)

    output = jnp.concatenate(o_per_qtile, axis=0)
    expected = _causal_attention_jnp(Q.arr, K.arr, V.arr)
    assert jnp.allclose(output, expected, atol=1e-5), \
        "Causal attention output does not numerically match reference."

    n_tiles = qctx.size // qctx_tile_size
    assert tiles_visited == n_tiles * (n_tiles + 1) // 2
    assert tiles_skipped == n_tiles * (n_tiles - 1) // 2
