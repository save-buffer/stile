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
    from stile.torch._core import TypedTorchTensor
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
