"""
TypedPallas — first kernels, structural verification + numerical sanity.

Each test runs the kernel under `interpret=True` so the dev loop is
local; the same trace lowers to CUDA/TPU on real hardware. The
verifier sees identical ASTs either way.
"""
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import pytest

import stile.jax as tjax
import stile.jax.pallas as tpl
from stile import dim, reset_stile


def test_scalar_multiply(reset):
    """The smallest possible typed-pallas kernel: load the input, scale
    by 2, store. Spec says `2 * TPN`. Verifier must accept and the
    runtime must match `2 * x`."""
    N = dim("TPN", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def kernel(x_ref : tpl.TypedRef, o_ref : tpl.TypedOutputRef):
        o_ref.assign(x_ref.load() * 2)

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * TPN", x.type.st, jnp.float32),
    )(x)

    assert isinstance(result, tjax.TypedJaxArray)
    assert jnp.allclose(result.arr, x.arr * 2)


def test_scalar_multiply_wrong_factor_rejected(reset):
    """Spec says `2 * TPN_W` but kernel multiplies by 3 — verifier
    rejects at `.assign(...)` time, before the store."""
    N = dim("TPN_W", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def kernel(x_ref, o_ref):
        o_ref.assign(x_ref.load() * 3)  # wrong factor

    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * TPN_W", x.type.st, jnp.float32),
    )
    with pytest.raises(ValueError, match="does not match spec"):
        runner(x)


def test_vector_add(reset):
    """Two distinct same-shape inputs, summed elementwise. The spec
    uses explicit `a:` / `b:` labels so the two tensors are treated as
    distinct leaves; the kernel inputs are constructed with matching
    `name=` so the spec-side and kernel-side identities align."""
    N = dim("VAN", 16)
    a = tjax.random.normal(jax.random.PRNGKey(0), N, name="a")
    b = tjax.random.normal(jax.random.PRNGKey(1), N, name="b")

    def kernel(a_ref : tpl.TypedRef,
               b_ref : tpl.TypedRef,
               o_ref : tpl.TypedOutputRef):
        o_ref.assign(a_ref.load() + b_ref.load())

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("a:VAN + b:VAN", a.type.st, jnp.float32),
    )(a, b)

    assert isinstance(result, tjax.TypedJaxArray)
    assert jnp.allclose(result.arr, a.arr + b.arr)


def test_vector_add_with_scaled_operands(reset):
    """`2*a + 3*b` with two distinct same-shape inputs."""
    N = dim("VANS", 16)
    a = tjax.random.normal(jax.random.PRNGKey(0), N, name="a")
    b = tjax.random.normal(jax.random.PRNGKey(1), N, name="b")

    def kernel(a_ref, b_ref, o_ref):
        o_ref.assign(a_ref.load() * 2 + b_ref.load() * 3)

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * a:VANS + 3 * b:VANS", a.type.st, jnp.float32),
    )(a, b)

    assert jnp.allclose(result.arr, a.arr * 2 + b.arr * 3)


def test_matmul(reset):
    """Non-tiled matmul: spec `(M N, N K -> M K)`. The kernel runs the
    full einsum inside a single Pallas invocation. Inputs have distinct
    dim signatures (`M N` vs `N K`), so they're separate leaves and
    the verifier sees a real contraction."""
    M, N, K = dim("M", 8), dim("N", 8), dim("K", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), M, N)
    b = tjax.random.normal(jax.random.PRNGKey(1), N, K)

    def kernel(a_ref, b_ref, o_ref):
        o_ref.assign(tjax.einsum(
            a_ref.load(), b_ref.load(),
            "M N, N K -> M K",
        ))

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("(M N, N K -> M K)", (M, K), jnp.float32),
    )(a, b)

    assert jnp.allclose(result.arr, a.arr @ b.arr, atol=1e-5)


def test_matmul_tiled(reset):
    """Tiled matmul: grid=(M//BM, K//BK), each invocation matmuls one
    `(BM, N)` tile of A by one `(N, BK)` tile of B into one `(BM, BK)`
    tile of the output. Stile derives each ref's `Type` from its
    BlockSpec — the M-axis of A and the K-axis of B come back sliced
    by the (symbolic) `program_id`. The per-block `assign(...)`
    certifies that the kernel's tile equals the spec's matmul
    restricted to the tile."""
    M, N, K = dim("M", 16), dim("N", 8), dim("K", 16)
    BM, BK = 8, 8
    a = tjax.random.normal(jax.random.PRNGKey(0), M, N, name="a")
    b = tjax.random.normal(jax.random.PRNGKey(1), N, K, name="b")

    def kernel(a_ref, b_ref, o_ref):
        o_ref.assign(tjax.einsum(
            a_ref.load(), b_ref.load(),
            "M N, N K -> M K",
        ))

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("(a:M N, b:N K -> M K)", (M, K), jnp.float32),
        grid=(M.size // BM, K.size // BK),
        in_specs=[
            pl.BlockSpec((BM, N.size), lambda m, k: (m, 0)),
            pl.BlockSpec((N.size, BK), lambda m, k: (0, k)),
        ],
        out_specs=pl.BlockSpec((BM, BK), lambda m, k: (m, k)),
    )(a, b)

    assert jnp.allclose(result.arr, a.arr @ b.arr, atol=1e-5)


def test_matmul_tiled_wrong_factor_rejected(reset):
    """Tiled kernel with the same buggy double-factor as the JAX-level
    test. Spec says `2 * matmul`; kernel multiplies BOTH operands by
    2, yielding 4× — verifier rejects per-block."""
    M, N, K = dim("M", 16), dim("N", 8), dim("K", 16)
    BM, BK = 8, 8
    a = tjax.random.normal(jax.random.PRNGKey(0), M, N, name="a")
    b = tjax.random.normal(jax.random.PRNGKey(1), N, K, name="b")

    def kernel(a_ref, b_ref, o_ref):
        o_ref.assign(tjax.einsum(
            a_ref.load() * 2, b_ref.load() * 2,  # bug: 4× total
            "M N, N K -> M K",
        ))

    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(
            "2 * (a:M N, b:N K -> M K)", (M, K), jnp.float32,
        ),
        grid=(M.size // BM, K.size // BK),
        in_specs=[
            pl.BlockSpec((BM, N.size), lambda m, k: (m, 0)),
            pl.BlockSpec((N.size, BK), lambda m, k: (0, k)),
        ],
        out_specs=pl.BlockSpec((BM, BK), lambda m, k: (m, k)),
    )
    with pytest.raises(ValueError, match="does not match spec"):
        runner(a, b)


def test_flash_attention(reset):
    """Real flash attention on Pallas: grid parallelizes over qctx
    tiles, the inner Python loop streams over nctx tiles using online
    softmax. The full QK matrix is never materialized; per-iteration
    aggregates are `running_max`, `running_l`, and `o`. The verifier
    merges each iteration's `Reduce(sum/max, dim=Sliced(nctx, ...))`
    into a single full-nctx `Reduce` and proves equivalence to
    `softmax[nctx](...)` via the same online-softmax convergence we
    have on the JAX backend."""
    qctx = dim("qctx", 16)
    nctx = dim("nctx", 16)
    dhead = dim("dhead", 16)
    BQ = 8
    BN = 8
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = tjax.random.normal(k1, qctx, dhead, name="Q")
    K = tjax.random.normal(k2, nctx, dhead, name="K")
    V = tjax.random.normal(k3, nctx, dhead, name="V")

    def kernel(q_ref, k_ref, v_ref, o_ref):
        q = q_ref.load()
        k_full = k_ref.load()
        v_full = v_ref.load()

        running_max = -jnp.inf
        running_l = 0
        o = 0

        for ictx in range(0, nctx.size, BN):
            k = k_full.slice(nctx, ictx, ictx + BN)
            v = v_full.slice(nctx, ictx, ictx + BN)
            qk_tile = tjax.einsum(
                q, k, "qctx dhead, nctx dhead -> qctx nctx",
            ) / tjax.sqrt(dhead.size)
            sliced_nctx = qk_tile.type.st[1]  # Sliced(nctx, ictx, ictx+BN)
            tile_max = qk_tile.max(nctx)
            logits = tjax.exp(
                qk_tile - tile_max.repeat(sliced_nctx).rearrange(qctx, nctx)
            )
            tile_l = logits.sum(nctx)

            new_max = tjax.maximum(tile_max, running_max)
            new_l = (
                tjax.exp(running_max - new_max) * running_l
                + tjax.exp(tile_max - new_max) * tile_l
            )
            v_proj = tjax.einsum(
                logits, v, "qctx nctx, nctx dhead -> qctx dhead",
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

        o_ref.assign(o)

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(
            "(softmax[nctx]((Q:qctx dhead, K:nctx dhead -> qctx nctx) / sqrt(16)), "
            "V:nctx dhead -> qctx dhead)",
            (qctx, dhead), jnp.float32,
        ),
        grid=(qctx.size // BQ,),
        in_specs=[
            pl.BlockSpec((BQ, dhead.size), lambda m: (m, 0)),       # Q tile
            pl.BlockSpec((nctx.size, dhead.size), lambda m: (0, 0)),  # K full
            pl.BlockSpec((nctx.size, dhead.size), lambda m: (0, 0)),  # V full
        ],
        out_specs=pl.BlockSpec((BQ, dhead.size), lambda m: (m, 0)),
    )(Q, K, V)

    # Numerical check against jnp reference.
    qk_full = jnp.einsum("qd,nd->qn", Q.arr, K.arr) / jnp.sqrt(dhead.size)
    softmax_full = jax.nn.softmax(qk_full, axis=-1)
    expected = jnp.einsum("qn,nd->qd", softmax_full, V.arr)
    assert jnp.allclose(result.arr, expected, atol=1e-5)


def test_flash_attention_rolled_inner(reset):
    """
    Flash attention with the inner nctx loop rolled via
    `tjax.fori_loop(..., invariant=...)` instead of Python-unrolled.
    The carry is `(m, l, o)` where `o` is the *un-normalized*
    numerator `sum(exp(qk - m_new) * V)` and `l` is the denominator
    `sum(exp(qk - m_new))` — both running aggregates against the
    latest running max. The final softmax-normalized output is `o / l`,
    computed once after the loop. Exercises the tile-aware invariant
    machinery: `typed_pallas_call` pushes the kernel's Sliced overrides
    onto an active-tile stack and `_fori_loop_with_invariant` runs
    `override_dims_in_type` on the parsed invariant Types so the body's
    Sliced qctx matches the invariant's.
    """
    qctx = dim("qctx", 16)
    nctx = dim("nctx", 16)
    dhead = dim("dhead", 16)
    BQ = 8
    BN = 8
    n_nctx_tiles = nctx.size // BN
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = tjax.random.normal(k1, qctx, dhead, name="Q")
    K = tjax.random.normal(k2, nctx, dhead, name="K")
    V = tjax.random.normal(k3, nctx, dhead, name="V")

    qk_str = (
        f"(Q:qctx dhead, K:nctx dhead -> qctx nctx) / sqrt({dhead.size})"
    )
    pred = f"nctx < {BN} * k"
    m_inv = f"max[nctx where {pred}]({qk_str})"
    l_inv = f"sum[nctx where {pred}](exp(({qk_str}) - {m_inv}))"
    o_inv = (
        f"sum[nctx where {pred}](exp(({qk_str}) - {m_inv}) * V:nctx dhead)"
    )

    def kernel(q_ref, k_ref, v_ref, o_ref):
        q = q_ref.load()
        k_full = k_ref.load()
        v_full = v_ref.load()

        def body(ki, state):
            m, l, o = state
            ictx = ki * BN
            k = k_full.slice(nctx, ictx, ictx + BN)
            v = v_full.slice(nctx, ictx, ictx + BN)
            qk_tile = tjax.einsum(
                q, k, "qctx dhead, nctx dhead -> qctx nctx",
            ) / tjax.sqrt(dhead.size)
            sliced_nctx = qk_tile.type.st[1]
            tile_max = qk_tile.max(nctx)
            new_max = tjax.maximum(tile_max, m)
            # `new_max` (not `tile_max`) in the exp so the per-iter
            # contribution to `l`/`o` is anchored on the running max —
            # matches the un-normalized invariant form.
            logits = tjax.exp(
                qk_tile - new_max.repeat(sliced_nctx).rearrange(qctx, nctx)
            )
            tile_l = logits.sum(nctx)
            new_l = tjax.exp(m - new_max) * l + tile_l
            v_proj = tjax.einsum(
                logits, v, "qctx nctx, nctx dhead -> qctx dhead",
            )
            new_o = (
                tjax.exp(m - new_max).repeat(dhead).rearrange(qctx, dhead) * o
                + v_proj
            )
            return (new_max, new_l, new_o)

        _, l_final, o_final = tjax.fori_loop(
            0, n_nctx_tiles, body,
            init_val=(-jnp.inf, 0.0, 0.0),
            invariant=(m_inv, l_inv, o_inv),
        )
        attn = o_final / l_final.repeat(dhead).rearrange(qctx, dhead)
        o_ref.assign(attn)

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(
            "(softmax[nctx]((Q:qctx dhead, K:nctx dhead -> qctx nctx) "
            f"/ sqrt({dhead.size})), V:nctx dhead -> qctx dhead)",
            (qctx, dhead), jnp.float32,
        ),
        grid=(qctx.size // BQ,),
        in_specs=[
            pl.BlockSpec((BQ, dhead.size), lambda m: (m, 0)),
            pl.BlockSpec((nctx.size, dhead.size), lambda m: (0, 0)),
            pl.BlockSpec((nctx.size, dhead.size), lambda m: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BQ, dhead.size), lambda m: (m, 0)),
    )(Q, K, V)

    qk_full = jnp.einsum("qd,nd->qn", Q.arr, K.arr) / jnp.sqrt(dhead.size)
    softmax_full = jax.nn.softmax(qk_full, axis=-1)
    expected = jnp.einsum("qn,nd->qd", softmax_full, V.arr)
    assert jnp.allclose(result.arr, expected, atol=1e-5)


def test_causal_flash_attention(reset):
    """Causal flash attention on Pallas: streaming online softmax over
    nctx tiles with a per-tile bias mask. Same shape as the dense
    streaming case but each tile adds `tjax.mask(...,
    "nctx <= qctx", 0.0, -jnp.inf)` before the softmax. The slice's
    qctx offset is symbolic at trace time (`pl.program_id(0) * BQ`)
    and the nctx tile offset is the Python-loop ictx; both flow
    through `loop_var_binding` and the dim_start machinery."""
    qctx = dim("qctx", 16)
    nctx = dim("nctx", 16)
    dhead = dim("dhead", 16)
    BQ = 8
    BN = 8
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = tjax.random.normal(k1, qctx, dhead, name="Q")
    K = tjax.random.normal(k2, nctx, dhead, name="K")
    V = tjax.random.normal(k3, nctx, dhead, name="V")

    def kernel(q_ref, k_ref, v_ref, o_ref):
        q = q_ref.load()
        k_full = k_ref.load()
        v_full = v_ref.load()

        running_max = -jnp.inf
        running_l = 0
        o = 0

        for ictx in range(0, nctx.size, BN):
            k = k_full.slice(nctx, ictx, ictx + BN)
            v = v_full.slice(nctx, ictx, ictx + BN)
            qk_tile = tjax.einsum(
                q, k, "qctx dhead, nctx dhead -> qctx nctx",
            ) / tjax.sqrt(dhead.size)
            sliced_nctx = qk_tile.type.st[1]
            # Per-tile causal bias: 0 inside `nctx <= qctx`, -inf outside.
            qk_tile = qk_tile + tjax.mask(
                qk_tile.type.st, "nctx <= qctx", 0.0, -jnp.inf,
            )
            tile_max = qk_tile.max(nctx)
            # Numerical fix for fully-masked rows: when every k in a tile
            # is causally invalid for some q, `tile_max[q]` is `-inf` and
            # `qk - tile_max = -inf - (-inf) = NaN`. Substitute `0` for
            # the subtraction — that row's contribution is `0` either
            # way (`exp(-inf - 0) = exp(-inf) = 0`), so the math is
            # unchanged. Type-level identity is preserved (jax-side
            # `where` doesn't alter the AST).
            tile_max_safe = tjax.TypedJaxArray(
                jnp.where(jnp.isinf(tile_max.arr), 0.0, tile_max.arr),
                tile_max.type,
            )
            logits = tjax.exp(
                qk_tile - tile_max_safe.repeat(sliced_nctx).rearrange(qctx, nctx)
            )
            tile_l = logits.sum(nctx)

            new_max = tjax.maximum(tile_max, running_max)
            new_l = (
                tjax.exp(running_max - new_max) * running_l
                + tjax.exp(tile_max - new_max) * tile_l
            )
            v_proj = tjax.einsum(
                logits, v, "qctx nctx, nctx dhead -> qctx dhead",
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

        o_ref.assign(o)

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(
            "(softmax[nctx where nctx <= qctx]"
            "((Q:qctx dhead, K:nctx dhead -> qctx nctx) / sqrt(16)), "
            "V:nctx dhead -> qctx dhead)",
            (qctx, dhead), jnp.float32,
        ),
        grid=(qctx.size // BQ,),
        in_specs=[
            pl.BlockSpec((BQ, dhead.size), lambda m: (m, 0)),
            pl.BlockSpec((nctx.size, dhead.size), lambda m: (0, 0)),
            pl.BlockSpec((nctx.size, dhead.size), lambda m: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BQ, dhead.size), lambda m: (m, 0)),
    )(Q, K, V)

    # Numerical check against jnp causal-attention reference.
    qk_full = jnp.einsum("qd,nd->qn", Q.arr, K.arr) / jnp.sqrt(dhead.size)
    q_idx = jnp.arange(qctx.size)[:, None]
    k_idx = jnp.arange(nctx.size)[None, :]
    qk_full = jnp.where(k_idx <= q_idx, qk_full, -jnp.inf)
    softmax_full = jax.nn.softmax(qk_full, axis=-1)
    expected = jnp.einsum("qn,nd->qd", softmax_full, V.arr)
    assert jnp.allclose(result.arr, expected, atol=1e-5)


def test_causal_decode_attention(reset):
    """Decode-step causal attention: a small number of new query
    tokens (`qctx`) attending against a longer K/V cache (`nctx`).
    Each new query at local position `q` corresponds to absolute
    position `q + offset` where `offset = nctx.size - qctx.size`. The
    diagonal of the attention matrix sits at the *right* of the strip:
    each new token attends to all cache tokens up to its own absolute
    index. We hardcode the offset into the predicate via an f-string —
    the parser's `Affine + int` grammar accepts it directly, so both
    the kernel-side `tjax.mask(...)` and the spec-side OutputSpec see
    the same constraint with the same literal."""
    qctx = dim("qctx", 8)    # 8 new tokens being decoded
    nctx = dim("nctx", 16)   # 16-token K/V cache (cache + new)
    dhead = dim("dhead", 16)
    BQ = 4
    qctx_offset = nctx.size - qctx.size
    predicate = f"nctx <= qctx + {qctx_offset}"

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = tjax.random.normal(k1, qctx, dhead, name="Q")
    K = tjax.random.normal(k2, nctx, dhead, name="K")
    V = tjax.random.normal(k3, nctx, dhead, name="V")

    def kernel(q_ref, k_ref, v_ref, o_ref):
        q = q_ref.load()
        k = k_ref.load()
        v = v_ref.load()
        qk = tjax.einsum(
            q, k, "qctx dhead, nctx dhead -> qctx nctx",
        ) / tjax.sqrt(dhead.size)
        qk = qk + tjax.mask(qk.type.st, predicate, 0.0, -jnp.inf)
        qk_max = qk.max(nctx)
        exp_qk = tjax.exp(qk - qk_max.repeat(nctx).rearrange(qctx, nctx))
        denom = exp_qk.sum(nctx)
        softmax_qk = exp_qk / denom.repeat(nctx).rearrange(qctx, nctx)
        v_proj = tjax.einsum(
            softmax_qk, v, "qctx nctx, nctx dhead -> qctx dhead",
        )
        o_ref.assign(v_proj)

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(
            f"(softmax[nctx where {predicate}]"
            f"((Q:qctx dhead, K:nctx dhead -> qctx nctx) / sqrt(16)), "
            f"V:nctx dhead -> qctx dhead)",
            (qctx, dhead), jnp.float32,
        ),
        grid=(qctx.size // BQ,),
        in_specs=[
            pl.BlockSpec((BQ, dhead.size), lambda m: (m, 0)),
            pl.BlockSpec((nctx.size, dhead.size), lambda m: (0, 0)),
            pl.BlockSpec((nctx.size, dhead.size), lambda m: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BQ, dhead.size), lambda m: (m, 0)),
    )(Q, K, V)

    # Numerical check: each query at local q ∈ [0, 8) corresponds to
    # absolute position q + 8; it attends to nctx ∈ [0, q + 8 + 1).
    qk_full = jnp.einsum("qd,nd->qn", Q.arr, K.arr) / jnp.sqrt(dhead.size)
    q_idx = jnp.arange(qctx.size)[:, None]
    k_idx = jnp.arange(nctx.size)[None, :]
    qk_full = jnp.where(k_idx <= q_idx + qctx_offset, qk_full, -jnp.inf)
    softmax_full = jax.nn.softmax(qk_full, axis=-1)
    expected = jnp.einsum("qn,nd->qd", softmax_full, V.arr)
    assert jnp.allclose(result.arr, expected, atol=1e-5)


def test_matmul_wrong_contracted_dim_rejected(reset):
    """Spec contracts N (the shared inner dim); kernel writes the
    wrong einsum string and contracts K instead — verifier rejects."""
    M, N, K = dim("M", 8), dim("N", 8), dim("K", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), M, N)
    b = tjax.random.normal(jax.random.PRNGKey(1), N, K)

    def kernel(a_ref, b_ref, o_ref):
        # Buggy: contracts the wrong dim, producing shape (M, N).
        o_ref.assign(tjax.einsum(
            a_ref.load(), b_ref.load(),
            "M N, N K -> M N",
        ))

    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(
            "(M N, N K -> M K)",
            (M, K), jnp.float32,
        ),
    )
    with pytest.raises(ValueError, match="does not match spec"):
        runner(a, b)
