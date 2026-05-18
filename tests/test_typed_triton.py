"""
TypedTriton — first kernel, structural verification + numerical sanity.

The user writes a kernel that looks like plain Triton — `tl.load` /
`tl.store` / arithmetic — and stile derives the typing by abstract-
interpreting the function's AST at decoration time. The verification
half runs locally without a GPU; the execution half is skipped if
Triton isn't importable.
"""
import pytest

import stile.triton as ttl
from stile import dim, reset_stile

try:
    import torch
    from stile.torch._core import TypedTorchTensor, runtime_index, tensor as ttensor
    from stile.type import Type, Tensor
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

if ttl.HAS_TRITON:
    import triton.language as tl
else:
    tl = None


@pytest.fixture
def reset():
    yield
    reset_stile()


_REQUIRES_TORCH = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
_REQUIRES_TRITON = pytest.mark.skipif(
    not ttl.HAS_TRITON, reason="Triton not installed",
)


@_REQUIRES_TRITON
@_REQUIRES_TORCH
def test_scalar_multiply_verification(reset):
    """
    Verification half only — runnable locally without a GPU. The
    `@ttl.jit` decorator reads the kernel's source, abstract-
    interprets the body, and checks the `tl.store`d value matches
    `2 * X:N`. No GPU work happens at decoration time.
    """
    N = dim("TTN", 128)
    BLOCK = 32

    @ttl.jit(
        spec="2 * X:TTN",
        inputs={"X_ptr": "X:TTN"},
        out_shape=(N,),
    )
    def double(X_ptr, o_ptr, BLOCK : tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        X = tl.load(X_ptr + offs, mask=mask)
        result = X * 2
        tl.store(o_ptr + offs, result, mask=mask)

    # Reaching here means abstract interpretation accepted the body
    # against the spec. (`TypedTritonKernel` is returned regardless of
    # whether Triton is actually available for execution.)
    assert isinstance(double, ttl.TypedTritonKernel)


@_REQUIRES_TRITON
@_REQUIRES_TORCH
def test_scalar_multiply(reset):
    """
    Full path: `@ttl.jit` verifies + emits the stripped `@triton.jit`
    function; `[grid](x)` launches and we numerically check `x * 2`.
    """
    N = dim("TTN", 128)
    BLOCK = 32

    @ttl.jit(
        spec="2 * X:TTN",
        inputs={"X_ptr": "X:TTN"},
        out_shape=(N,),
        out_dtype=torch.float32,
    )
    def double(X_ptr, o_ptr, BLOCK : tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        X = tl.load(X_ptr + offs, mask=mask)
        result = X * 2
        tl.store(o_ptr + offs, result, mask=mask)

    x_arr = torch.randn(N.size, device="cuda")
    x_type = Type(st=(N,), et=Tensor(dims=(N,), name="X"))
    x = TypedTorchTensor(x_arr, x_type)

    result = double[(N.size // BLOCK,)](x, BLOCK=BLOCK)
    assert torch.allclose(result.tensor, x_arr * 2, atol=1e-5)


@_REQUIRES_TRITON
@_REQUIRES_TORCH
def test_scalar_multiply_slice_syntax(reset):
    """
    Same kernel, but using `ttl.load(ptr, DIM[lo:hi])` /
    `ttl.store(ptr, value, DIM[lo:hi])` — the stile-flavored slice
    syntax that mirrors `.slice(dim, lo, hi)`. The source-level
    rewriter turns these into raw `tl.load`/`tl.store` with offsets
    + mask before handing the function to `@triton.jit`.
    """
    N = dim("TTN", 128)
    BLOCK = 32

    @ttl.jit(
        spec="2 * X:TTN",
        inputs={"X_ptr": "X:TTN"},
        out_shape=(N,),
        out_dtype=torch.float32,
    )
    def double(X_ptr, o_ptr, BLOCK : tl.constexpr):
        pid = tl.program_id(0)
        X = ttl.load(X_ptr, N[pid * BLOCK : (pid + 1) * BLOCK])
        result = X * 2
        ttl.store(o_ptr, result, N[pid * BLOCK : (pid + 1) * BLOCK])

    x_arr = torch.randn(N.size, device="cuda")
    x_type = Type(st=(N,), et=Tensor(dims=(N,), name="X"))
    x = TypedTorchTensor(x_arr, x_type)

    result = double[(N.size // BLOCK,)](x, BLOCK=BLOCK)
    assert torch.allclose(result.tensor, x_arr * 2, atol=1e-5)


@_REQUIRES_TRITON
@_REQUIRES_TORCH
def test_matmul(reset):
    """
    Simple non-K-tiled matmul. Each program computes one BLOCK_M ×
    BLOCK_N tile of C from a full M × K column-strip of A and full
    K × N row-strip of B. Demonstrates multi-dim `ttl.load`/`ttl.store`
    with the `DIM[lo:hi]` syntax plus `tl.dot` interpretation.
    """
    M = dim("M", 32)
    N = dim("N", 32)
    K = dim("K", 32)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = K.size  # whole K in one tile

    @ttl.jit(
        spec="(A:M K, B:K N -> M N)",
        inputs={"A_ptr": "A:M K", "B_ptr": "B:K N"},
        out_shape=(M, N),
        out_dtype=torch.float32,
    )
    def matmul(
        A_ptr, B_ptr, C_ptr,
        BLOCK_M : tl.constexpr, BLOCK_N : tl.constexpr,
        BLOCK_K : tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        a = ttl.load(
            A_ptr,
            M[pid_m * BLOCK_M : (pid_m + 1) * BLOCK_M],
            K[0 : BLOCK_K],
        )
        b = ttl.load(
            B_ptr,
            K[0 : BLOCK_K],
            N[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N],
        )
        c = tl.dot(a, b)
        ttl.store(
            C_ptr, c,
            M[pid_m * BLOCK_M : (pid_m + 1) * BLOCK_M],
            N[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N],
        )

    a_arr = torch.randn(M.size, K.size, device="cuda")
    b_arr = torch.randn(K.size, N.size, device="cuda")
    a = TypedTorchTensor(
        a_arr, Type(st=(M, K), et=Tensor(dims=(M, K), name="A")),
    )
    b = TypedTorchTensor(
        b_arr, Type(st=(K, N), et=Tensor(dims=(K, N), name="B")),
    )
    result = matmul[(M.size // BLOCK_M, N.size // BLOCK_N)](
        a, b,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    # `tl.dot` defaults to tf32, so the result differs from PyTorch's
    # fp32 matmul by a small absolute drift (verified bit-for-bit
    # equal when `input_precision="ieee"` is passed). Tolerance
    # reflects the tf32 path.
    expected = a_arr @ b_arr
    assert torch.allclose(result.tensor, expected, atol=2e-2, rtol=1e-3)


@_REQUIRES_TRITON
@_REQUIRES_TORCH
def test_matmul_tiled_k(reset):
    """
    Tiled matmul with an inner K loop. K = 64, BLOCK_K = 32, so the
    `for ki in range(0, K, BLOCK_K):` body runs twice; the accumulator
    sums the two `tl.dot` contributions. Verification unrolls the
    loop concretely, builds per-iteration sliced load Types, and
    relies on stile's tile-merge to fold `sum_over_tiles
    einsum(A:Sliced(K, t*BLOCK_K, (t+1)*BLOCK_K), …)` into the full-K
    einsum that the spec declares.
    """
    M = dim("M", 32)
    N = dim("N", 32)
    K = dim("K", 64)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32

    @ttl.jit(
        spec="(A:M K, B:K N -> M N)",
        inputs={"A_ptr": "A:M K", "B_ptr": "B:K N"},
        out_shape=(M, N),
        out_dtype=torch.float32,
        consts={"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
    )
    def matmul(
        A_ptr, B_ptr, C_ptr,
        BLOCK_M : tl.constexpr, BLOCK_N : tl.constexpr,
        BLOCK_K : tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        acc = ttl.zeros(
            M[pid_m * BLOCK_M : (pid_m + 1) * BLOCK_M],
            N[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N],
        )
        for ki in range(0, K, BLOCK_K):
            a = ttl.load(
                A_ptr,
                M[pid_m * BLOCK_M : (pid_m + 1) * BLOCK_M],
                K[ki : ki + BLOCK_K],
            )
            b = ttl.load(
                B_ptr,
                K[ki : ki + BLOCK_K],
                N[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N],
            )
            acc += tl.dot(a, b)
        ttl.store(
            C_ptr, acc,
            M[pid_m * BLOCK_M : (pid_m + 1) * BLOCK_M],
            N[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N],
        )

    a_arr = torch.randn(M.size, K.size, device="cuda")
    b_arr = torch.randn(K.size, N.size, device="cuda")
    a = TypedTorchTensor(
        a_arr, Type(st=(M, K), et=Tensor(dims=(M, K), name="A")),
    )
    b = TypedTorchTensor(
        b_arr, Type(st=(K, N), et=Tensor(dims=(K, N), name="B")),
    )
    result = matmul[(M.size // BLOCK_M, N.size // BLOCK_N)](a, b)
    expected = a_arr @ b_arr
    assert torch.allclose(result.tensor, expected, atol=2e-2, rtol=1e-3)


@_REQUIRES_TRITON
@_REQUIRES_TORCH
def test_flash_attention(reset):
    """
    Non-causal flash attention via the Triton frontend. Outer grid
    tiles over qctx (one program per BQ-row block); inner loop tiles
    over nctx with the standard online-softmax (m, l, o) accumulator
    triple. The invariants on each accumulator are the same shape we
    verified jax-side in `test_loop_invariants.test_flash_attention_via_invariant`,
    minus the causal predicate.
    """
    qctx = dim("qctx", 32)
    nctx = dim("nctx", 32)
    dhead = dim("dhead", 16)
    BQ = 16
    BN = 16
    BD = dhead.size

    qk_spec = "(Q:qctx dhead, K:nctx dhead -> qctx nctx)"
    m_inv = f"max[nctx where nctx < {BN} * ki]{qk_spec}"
    l_inv = f"sum[nctx where nctx < {BN} * ki](exp({qk_spec} - {m_inv}))"
    o_inv = (
        f"sum[nctx where nctx < {BN} * ki]"
        f"(exp({qk_spec} - {m_inv}) * V:nctx dhead)"
    )

    @ttl.jit(
        spec=f"(softmax[nctx]({qk_spec}), V:nctx dhead -> qctx dhead)",
        inputs={
            "Q_ptr": "Q:qctx dhead",
            "K_ptr": "K:nctx dhead",
            "V_ptr": "V:nctx dhead",
        },
        out_shape=(qctx, dhead),
        out_dtype=torch.float32,
        consts={"BQ": BQ, "BN": BN, "BD": BD},
    )
    def attn(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        BQ: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        q = ttl.load(
            Q_ptr,
            qctx[pid_m * BQ : (pid_m + 1) * BQ],
            dhead[0 : BD],
        )
        m = ttl.full(qctx[pid_m * BQ : (pid_m + 1) * BQ], value=float("-inf"))
        l = ttl.zeros(qctx[pid_m * BQ : (pid_m + 1) * BQ])
        o = ttl.zeros(
            qctx[pid_m * BQ : (pid_m + 1) * BQ],
            dhead[0 : BD],
        )
        for ki in ttl.range(
            0, nctx // BN,
            invariant={"m": m_inv, "l": l_inv, "o": o_inv},
        ):
            k = ttl.load(
                K_ptr,
                nctx[ki * BN : (ki + 1) * BN],
                dhead[0 : BD],
            )
            v = ttl.load(
                V_ptr,
                nctx[ki * BN : (ki + 1) * BN],
                dhead[0 : BD],
            )
            qk = tl.dot(q, tl.trans(k))
            tile_max = tl.max(qk, axis=1)
            new_max = tl.maximum(m, tile_max)
            logits = tl.exp(qk - new_max[:, None])
            tile_l = tl.sum(logits, axis=1)
            new_l = tl.exp(m - new_max) * l + tile_l
            v_proj = tl.dot(logits, v)
            new_o = tl.exp(m - new_max)[:, None] * o + v_proj
            m = new_max
            l = new_l
            o = new_o
        attn_out = o / l[:, None]
        ttl.store(
            O_ptr, attn_out,
            qctx[pid_m * BQ : (pid_m + 1) * BQ],
            dhead[0 : BD],
        )

    k1, k2, k3 = torch.randn(qctx.size, dhead.size, device="cuda"), torch.randn(nctx.size, dhead.size, device="cuda"), torch.randn(nctx.size, dhead.size, device="cuda")
    Q = TypedTorchTensor(k1, Type(st=(qctx, dhead), et=Tensor(dims=(qctx, dhead), name="Q")))
    K = TypedTorchTensor(k2, Type(st=(nctx, dhead), et=Tensor(dims=(nctx, dhead), name="K")))
    V = TypedTorchTensor(k3, Type(st=(nctx, dhead), et=Tensor(dims=(nctx, dhead), name="V")))
    result = attn[(qctx.size // BQ,)](Q, K, V)

    qk_full = k1 @ k2.T
    softmax_full = torch.softmax(qk_full, dim=-1)
    expected = softmax_full @ k3
    assert torch.allclose(result.tensor, expected, atol=2e-2, rtol=1e-3)


@_REQUIRES_TRITON
@_REQUIRES_TORCH
def test_flash_attention_causal(reset):
    """
    Causal flash attention. Same online-softmax (m, l, o) loop as the
    non-causal version, but each per-tile qk gets a bias mask
    (`0` inside the causal region, `-inf` outside) before the softmax.
    The invariant carries the conjunction `nctx < BN*ki && nctx <= qctx`,
    matching the spec's `softmax[nctx where nctx <= qctx]`.
    """
    qctx = dim("qctx", 32)
    nctx = dim("nctx", 32)
    dhead = dim("dhead", 16)
    BQ = 16
    BN = 16
    BD = dhead.size

    qk_spec = "(Q:qctx dhead, K:nctx dhead -> qctx nctx)"
    pred = f"nctx < {BN} * ki && nctx <= qctx"
    m_inv = f"max[nctx where {pred}]{qk_spec}"
    l_inv = f"sum[nctx where {pred}](exp({qk_spec} - {m_inv}))"
    o_inv = (
        f"sum[nctx where {pred}]"
        f"(exp({qk_spec} - {m_inv}) * V:nctx dhead)"
    )

    @ttl.jit(
        spec=(
            f"(softmax[nctx where nctx <= qctx]({qk_spec}), "
            f"V:nctx dhead -> qctx dhead)"
        ),
        inputs={
            "Q_ptr": "Q:qctx dhead",
            "K_ptr": "K:nctx dhead",
            "V_ptr": "V:nctx dhead",
        },
        out_shape=(qctx, dhead),
        out_dtype=torch.float32,
        consts={"BQ": BQ, "BN": BN, "BD": BD},
    )
    def attn(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        BQ: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        q = ttl.load(
            Q_ptr,
            qctx[pid_m * BQ : (pid_m + 1) * BQ],
            dhead[0 : BD],
        )
        m = ttl.full(qctx[pid_m * BQ : (pid_m + 1) * BQ], value=float("-inf"))
        l = ttl.zeros(qctx[pid_m * BQ : (pid_m + 1) * BQ])
        o = ttl.zeros(
            qctx[pid_m * BQ : (pid_m + 1) * BQ],
            dhead[0 : BD],
        )
        for ki in ttl.range(
            0, nctx // BN,
            invariant={"m": m_inv, "l": l_inv, "o": o_inv},
        ):
            k = ttl.load(
                K_ptr,
                nctx[ki * BN : (ki + 1) * BN],
                dhead[0 : BD],
            )
            v = ttl.load(
                V_ptr,
                nctx[ki * BN : (ki + 1) * BN],
                dhead[0 : BD],
            )
            qk = tl.dot(q, tl.trans(k))
            qk = qk + ttl.mask(
                qctx[pid_m * BQ : (pid_m + 1) * BQ],
                nctx[ki * BN : (ki + 1) * BN],
                predicate="nctx <= qctx",
                on=0.0, off=float("-inf"),
            )
            tile_max = tl.max(qk, axis=1)
            new_max = tl.maximum(m, tile_max)
            logits = tl.exp(qk - new_max[:, None])
            tile_l = tl.sum(logits, axis=1)
            new_l = tl.exp(m - new_max) * l + tile_l
            v_proj = tl.dot(logits, v)
            new_o = tl.exp(m - new_max)[:, None] * o + v_proj
            m = new_max
            l = new_l
            o = new_o
        attn_out = o / l[:, None]
        ttl.store(
            O_ptr, attn_out,
            qctx[pid_m * BQ : (pid_m + 1) * BQ],
            dhead[0 : BD],
        )

    Q_arr = torch.randn(qctx.size, dhead.size, device="cuda")
    K_arr = torch.randn(nctx.size, dhead.size, device="cuda")
    V_arr = torch.randn(nctx.size, dhead.size, device="cuda")
    Q = TypedTorchTensor(Q_arr, Type(st=(qctx, dhead), et=Tensor(dims=(qctx, dhead), name="Q")))
    K = TypedTorchTensor(K_arr, Type(st=(nctx, dhead), et=Tensor(dims=(nctx, dhead), name="K")))
    V = TypedTorchTensor(V_arr, Type(st=(nctx, dhead), et=Tensor(dims=(nctx, dhead), name="V")))
    result = attn[(qctx.size // BQ,)](Q, K, V)

    # Reference: causal-masked softmax attention.
    qk_full = Q_arr @ K_arr.T
    q_idx = torch.arange(qctx.size, device="cuda")[:, None]
    k_idx = torch.arange(nctx.size, device="cuda")[None, :]
    qk_full = qk_full.masked_fill(k_idx > q_idx, float("-inf"))
    expected = torch.softmax(qk_full, dim=-1) @ V_arr
    assert torch.allclose(result.tensor, expected, atol=2e-2, rtol=1e-3)


@_REQUIRES_TRITON
@_REQUIRES_TORCH
def test_rope(reset):
    """
    Rotary position embedding via the gather formulation (LLaMA's
    rotate-half trick). The 'rotation' is a permutation + sign-flip
    on `dhead`:
        σ = [HD, HD+1, …, D−1, 0, 1, …, HD−1]   (cyclic shift by HD)
        sign = [−1]·HD ⧺ [+1]·HD
    and `rope(x, θ) = x · cos(θ) + sign · x.gather(dhead, σ) · sin(θ)`.

    The kernel reads `rope_perm` and `sign_mask` as opaque named
    tensors; the verifier matches them structurally against the same
    names in the spec.
    """
    n_tokens = dim("n_tokens", 8)
    dhead = dim("dhead", 8)
    HD = dhead.size // 2
    BS = 4

    X_arr = torch.randn(n_tokens.size, dhead.size, device="cuda")
    cos_arr = torch.randn(n_tokens.size, dhead.size, device="cuda")
    sin_arr = torch.randn(n_tokens.size, dhead.size, device="cuda")
    perm_arr = torch.cat([
        torch.arange(HD, dhead.size, device="cuda"),
        torch.arange(0, HD, device="cuda"),
    ]).to(torch.int32)
    sign_arr = torch.cat([
        -torch.ones(HD, device="cuda"),
        torch.ones(HD, device="cuda"),
    ]).to(torch.float32)

    X = ttensor(X_arr, n_tokens, dhead, name="X")
    cos_table = ttensor(cos_arr, n_tokens, dhead, name="cos_table")
    sin_table = ttensor(sin_arr, n_tokens, dhead, name="sin_table")
    rope_perm = runtime_index(
        "rope_perm", dhead, values_in=dhead, permutation=True, arr=perm_arr,
    )
    sign_mask = ttensor(sign_arr, dhead, name="sign_mask")

    @ttl.jit(
        spec="(X:n_tokens dhead * cos_table:n_tokens dhead + "
             "gather[dhead](X:n_tokens dhead, rope_perm:dhead) "
             "* sign_mask:dhead * sin_table:n_tokens dhead)",
        inputs={
            "X_ptr":    "X:n_tokens dhead",
            "cos_ptr":  "cos_table:n_tokens dhead",
            "sin_ptr":  "sin_table:n_tokens dhead",
            "perm_ptr": "rope_perm:dhead",
            "sign_ptr": "sign_mask:dhead",
        },
        out_shape=(n_tokens, dhead),
        out_dtype=torch.float32,
        consts={"BS": BS, "D": dhead.size},
    )
    def rope(
        X_ptr, cos_ptr, sin_ptr, perm_ptr, sign_ptr, O_ptr,
        BS: tl.constexpr, D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        x = ttl.load(
            X_ptr,
            n_tokens[pid * BS : (pid + 1) * BS],
            dhead[0 : D],
        )
        cos = ttl.load(
            cos_ptr,
            n_tokens[pid * BS : (pid + 1) * BS],
            dhead[0 : D],
        )
        sin = ttl.load(
            sin_ptr,
            n_tokens[pid * BS : (pid + 1) * BS],
            dhead[0 : D],
        )
        perm = ttl.load(perm_ptr, dhead[0 : D])
        sign = ttl.load(sign_ptr, dhead[0 : D])
        rotated = ttl.gather(x, dhead, perm)
        out = x * cos + rotated * sign * sin
        ttl.store(
            O_ptr, out,
            n_tokens[pid * BS : (pid + 1) * BS],
            dhead[0 : D],
        )

    result = rope[(n_tokens.size // BS,)](
        X, cos_table, sin_table, rope_perm, sign_mask,
    )

    x_lo = X_arr[..., :HD]
    x_hi = X_arr[..., HD:]
    rotated_arr = torch.cat([-x_hi, x_lo], dim=-1)
    expected = X_arr * cos_arr + rotated_arr * sin_arr
    assert torch.allclose(result.tensor, expected, atol=1e-5)


@_REQUIRES_TRITON
@_REQUIRES_TORCH
def test_flash_attention_shifted_causal(reset):
    """
    Shifted-causal flash attention: each query at position q can attend
    to keys at positions `[0, q + OFFSET]` (rather than `[0, q]`).
    Exercises the offset path in the runtime mask rewriter — the
    predicate is `nctx <= qctx + OFFSET`, parameterized via an f-string
    that resolves at decoration time.
    """
    qctx = dim("qctx", 32)
    nctx = dim("nctx", 32)
    dhead = dim("dhead", 16)
    OFFSET = 4
    BQ = 16
    BN = 16
    BD = dhead.size

    qk_spec = "(Q:qctx dhead, K:nctx dhead -> qctx nctx)"
    pred = f"nctx < {BN} * ki && nctx <= qctx + {OFFSET}"
    m_inv = f"max[nctx where {pred}]{qk_spec}"
    l_inv = f"sum[nctx where {pred}](exp({qk_spec} - {m_inv}))"
    o_inv = (
        f"sum[nctx where {pred}]"
        f"(exp({qk_spec} - {m_inv}) * V:nctx dhead)"
    )

    @ttl.jit(
        spec=(
            f"(softmax[nctx where nctx <= qctx + {OFFSET}]({qk_spec}), "
            f"V:nctx dhead -> qctx dhead)"
        ),
        inputs={
            "Q_ptr": "Q:qctx dhead",
            "K_ptr": "K:nctx dhead",
            "V_ptr": "V:nctx dhead",
        },
        out_shape=(qctx, dhead),
        out_dtype=torch.float32,
        consts={"BQ": BQ, "BN": BN, "BD": BD},
    )
    def attn(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        BQ: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        q = ttl.load(
            Q_ptr,
            qctx[pid_m * BQ : (pid_m + 1) * BQ],
            dhead[0 : BD],
        )
        m = ttl.full(qctx[pid_m * BQ : (pid_m + 1) * BQ], value=float("-inf"))
        l = ttl.zeros(qctx[pid_m * BQ : (pid_m + 1) * BQ])
        o = ttl.zeros(
            qctx[pid_m * BQ : (pid_m + 1) * BQ],
            dhead[0 : BD],
        )
        for ki in ttl.range(
            0, nctx // BN,
            invariant={"m": m_inv, "l": l_inv, "o": o_inv},
        ):
            k = ttl.load(
                K_ptr,
                nctx[ki * BN : (ki + 1) * BN],
                dhead[0 : BD],
            )
            v = ttl.load(
                V_ptr,
                nctx[ki * BN : (ki + 1) * BN],
                dhead[0 : BD],
            )
            qk = tl.dot(q, tl.trans(k))
            qk = qk + ttl.mask(
                qctx[pid_m * BQ : (pid_m + 1) * BQ],
                nctx[ki * BN : (ki + 1) * BN],
                predicate=f"nctx <= qctx + {OFFSET}",
                on=0.0, off=float("-inf"),
            )
            tile_max = tl.max(qk, axis=1)
            new_max = tl.maximum(m, tile_max)
            logits = tl.exp(qk - new_max[:, None])
            tile_l = tl.sum(logits, axis=1)
            new_l = tl.exp(m - new_max) * l + tile_l
            v_proj = tl.dot(logits, v)
            new_o = tl.exp(m - new_max)[:, None] * o + v_proj
            m = new_max
            l = new_l
            o = new_o
        attn_out = o / l[:, None]
        ttl.store(
            O_ptr, attn_out,
            qctx[pid_m * BQ : (pid_m + 1) * BQ],
            dhead[0 : BD],
        )

    Q_arr = torch.randn(qctx.size, dhead.size, device="cuda")
    K_arr = torch.randn(nctx.size, dhead.size, device="cuda")
    V_arr = torch.randn(nctx.size, dhead.size, device="cuda")
    Q = TypedTorchTensor(Q_arr, Type(st=(qctx, dhead), et=Tensor(dims=(qctx, dhead), name="Q")))
    K = TypedTorchTensor(K_arr, Type(st=(nctx, dhead), et=Tensor(dims=(nctx, dhead), name="K")))
    V = TypedTorchTensor(V_arr, Type(st=(nctx, dhead), et=Tensor(dims=(nctx, dhead), name="V")))
    result = attn[(qctx.size // BQ,)](Q, K, V)

    # Reference: shifted-causal softmax attention.
    qk_full = Q_arr @ K_arr.T
    q_idx = torch.arange(qctx.size, device="cuda")[:, None]
    k_idx = torch.arange(nctx.size, device="cuda")[None, :]
    qk_full = qk_full.masked_fill(k_idx > q_idx + OFFSET, float("-inf"))
    expected = torch.softmax(qk_full, dim=-1) @ V_arr
    assert torch.allclose(result.tensor, expected, atol=2e-2, rtol=1e-3)


@_REQUIRES_TRITON
@_REQUIRES_TORCH
def test_moe_per_token_dispatch(reset):
    """
    Simplest MoE: each token's output is its assigned expert's matmul.
    `Y[t, :] = W[expert_id[t], :, :] @ X[t, :]`.

    Kernel loads the (small) full weight tensor `W:n_experts d_in d_out`
    once per program, then gathers along `n_experts` using the per-token
    `expert_id` slice. The gather output is `(BT, d_in, d_out)` — the
    per-token weight matrix — which we einsum-multiply against the
    `(BT, d_in)` X tile by broadcasting and summing over `d_in`.
    """
    n_tokens = dim("n_tokens", 16)
    d_in = dim("d_in", 4)
    d_out = dim("d_out", 8)
    n_experts = dim("n_experts", 4)
    BT = 8

    X_arr = torch.randn(n_tokens.size, d_in.size, device="cuda")
    W_arr = torch.randn(
        n_experts.size, d_in.size, d_out.size, device="cuda",
    )
    eid_arr = torch.randint(
        0, n_experts.size, (n_tokens.size,), device="cuda",
    ).to(torch.int32)

    X = ttensor(X_arr, n_tokens, d_in, name="X")
    W = ttensor(W_arr, n_experts, d_in, d_out, name="W")
    expert_id = runtime_index(
        "expert_id", n_tokens, values_in=n_experts, arr=eid_arr,
    )

    @ttl.jit(
        spec="(gather[n_experts](W:n_experts d_in d_out, expert_id:n_tokens), "
             "X:n_tokens d_in -> n_tokens d_out)",
        inputs={
            "X_ptr":   "X:n_tokens d_in",
            "W_ptr":   "W:n_experts d_in d_out",
            "eid_ptr": "expert_id:n_tokens",
        },
        out_shape=(n_tokens, d_out),
        out_dtype=torch.float32,
        consts={
            "BT": BT, "DI": d_in.size,
            "DO": d_out.size, "NE": n_experts.size,
        },
    )
    def moe(
        X_ptr, W_ptr, eid_ptr, O_ptr,
        BT: tl.constexpr, DI: tl.constexpr,
        DO: tl.constexpr, NE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        x = ttl.load(
            X_ptr,
            n_tokens[pid * BT : (pid + 1) * BT],
            d_in[0 : DI],
        )
        eid = ttl.load(eid_ptr, n_tokens[pid * BT : (pid + 1) * BT])
        W = ttl.load(
            W_ptr,
            n_experts[0 : NE],
            d_in[0 : DI],
            d_out[0 : DO],
        )
        W_per_token = ttl.gather(W, n_experts, eid)
        prod = x[:, :, None] * W_per_token
        y = tl.sum(prod, axis=1)
        ttl.store(
            O_ptr, y,
            n_tokens[pid * BT : (pid + 1) * BT],
            d_out[0 : DO],
        )

    result = moe[(n_tokens.size // BT,)](X, W, expert_id)

    expected = torch.einsum(
        "nd,nde->ne", X_arr, W_arr[eid_arr.to(torch.int64)],
    )
    assert torch.allclose(result.tensor, expected, atol=1e-4)


@_REQUIRES_TRITON
@_REQUIRES_TORCH
def test_moe_tiled(reset):
    """
    Tiled MoE — same per-token-dispatch structure, but the expert
    weight tensor is now big enough that it can't be loaded whole.
    The grid tiles tokens × d_out, and an inner `ttl.range` loop walks
    d_in. The loop-invariant says `acc` equals the partial einsum
    restricted to `d_in < BD_IN * ki`; at the inductive end (ki =
    d_in/BD_IN) that's the full einsum the spec declares.
    """
    n_tokens = dim("n_tokens", 32)
    d_in = dim("d_in", 64)
    d_out = dim("d_out", 64)
    n_experts = dim("n_experts", 4)
    BT = 16
    BD_IN = 16
    BD_OUT = 16

    X_arr = torch.randn(n_tokens.size, d_in.size, device="cuda")
    W_arr = torch.randn(
        n_experts.size, d_in.size, d_out.size, device="cuda",
    )
    eid_arr = torch.randint(
        0, n_experts.size, (n_tokens.size,), device="cuda",
    ).to(torch.int32)

    X = ttensor(X_arr, n_tokens, d_in, name="X")
    W = ttensor(W_arr, n_experts, d_in, d_out, name="W")
    expert_id = runtime_index(
        "expert_id", n_tokens, values_in=n_experts, arr=eid_arr,
    )

    inv = (
        f"sum[d_in where d_in < {BD_IN} * ki]("
        f"gather[n_experts](W:n_experts d_in d_out, expert_id:n_tokens) "
        f"* X:n_tokens d_in)"
    )

    @ttl.jit(
        spec="(gather[n_experts](W:n_experts d_in d_out, expert_id:n_tokens), "
             "X:n_tokens d_in -> n_tokens d_out)",
        inputs={
            "X_ptr":   "X:n_tokens d_in",
            "W_ptr":   "W:n_experts d_in d_out",
            "eid_ptr": "expert_id:n_tokens",
        },
        out_shape=(n_tokens, d_out),
        out_dtype=torch.float32,
        consts={
            "BT": BT, "BD_IN": BD_IN, "BD_OUT": BD_OUT,
            "NE": n_experts.size,
        },
    )
    def moe(
        X_ptr, W_ptr, eid_ptr, O_ptr,
        BT: tl.constexpr, BD_IN: tl.constexpr,
        BD_OUT: tl.constexpr, NE: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_o = tl.program_id(1)
        eid = ttl.load(eid_ptr, n_tokens[pid_t * BT : (pid_t + 1) * BT])
        acc = ttl.zeros(
            n_tokens[pid_t * BT : (pid_t + 1) * BT],
            d_out[pid_o * BD_OUT : (pid_o + 1) * BD_OUT],
        )
        for ki in ttl.range(
            0, d_in // BD_IN, invariant={"acc": inv},
        ):
            x = ttl.load(
                X_ptr,
                n_tokens[pid_t * BT : (pid_t + 1) * BT],
                d_in[ki * BD_IN : (ki + 1) * BD_IN],
            )
            W = ttl.load(
                W_ptr,
                n_experts[0 : NE],
                d_in[ki * BD_IN : (ki + 1) * BD_IN],
                d_out[pid_o * BD_OUT : (pid_o + 1) * BD_OUT],
            )
            W_per_token = ttl.gather(W, n_experts, eid)
            prod = x[:, :, None] * W_per_token
            acc += tl.sum(prod, axis=1)
        ttl.store(
            O_ptr, acc,
            n_tokens[pid_t * BT : (pid_t + 1) * BT],
            d_out[pid_o * BD_OUT : (pid_o + 1) * BD_OUT],
        )

    result = moe[(n_tokens.size // BT, d_out.size // BD_OUT)](
        X, W, expert_id,
    )

    expected = torch.einsum(
        "nd,nde->ne", X_arr, W_arr[eid_arr.to(torch.int64)],
    )
    assert torch.allclose(result.tensor, expected, atol=1e-4)


@_REQUIRES_TRITON
@_REQUIRES_TORCH
def test_matmul_with_invariant(reset):
    """
    Same tiled matmul, but the K-loop is wrapped in `ttl.range(...,
    invariant={"acc": …})`. The verifier discharges the loop with
    Hoare-style base + inductive instead of unrolling — verification
    cost is independent of the trip count, and the bounds may be
    symbolic. At emit time the `ttl.range` reduces to plain `range`
    so Triton sees ordinary Python.
    """
    M = dim("M", 32)
    N = dim("N", 32)
    K = dim("K", 64)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32

    @ttl.jit(
        spec="(A:M K, B:K N -> M N)",
        inputs={"A_ptr": "A:M K", "B_ptr": "B:K N"},
        out_shape=(M, N),
        out_dtype=torch.float32,
        consts={"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
    )
    def matmul(
        A_ptr, B_ptr, C_ptr,
        BLOCK_M : tl.constexpr, BLOCK_N : tl.constexpr,
        BLOCK_K : tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        acc = ttl.zeros(
            M[pid_m * BLOCK_M : (pid_m + 1) * BLOCK_M],
            N[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N],
        )
        for ki in ttl.range(
            0, K // BLOCK_K,
            invariant={
                "acc": f"sum[K where K < {BLOCK_K} * ki]("
                       f"A:M K * B:K N)",
            },
        ):
            a = ttl.load(
                A_ptr,
                M[pid_m * BLOCK_M : (pid_m + 1) * BLOCK_M],
                K[ki * BLOCK_K : (ki + 1) * BLOCK_K],
            )
            b = ttl.load(
                B_ptr,
                K[ki * BLOCK_K : (ki + 1) * BLOCK_K],
                N[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N],
            )
            acc += tl.dot(a, b)
        ttl.store(
            C_ptr, acc,
            M[pid_m * BLOCK_M : (pid_m + 1) * BLOCK_M],
            N[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N],
        )

    a_arr = torch.randn(M.size, K.size, device="cuda")
    b_arr = torch.randn(K.size, N.size, device="cuda")
    a = TypedTorchTensor(
        a_arr, Type(st=(M, K), et=Tensor(dims=(M, K), name="A")),
    )
    b = TypedTorchTensor(
        b_arr, Type(st=(K, N), et=Tensor(dims=(K, N), name="B")),
    )
    result = matmul[(M.size // BLOCK_M, N.size // BLOCK_N)](a, b)
    expected = a_arr @ b_arr
    assert torch.allclose(result.tensor, expected, atol=2e-2, rtol=1e-3)
