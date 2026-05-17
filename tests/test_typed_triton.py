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
