"""
Cross-framework reference verification.

Instead of a spec-language string, the expected output of a typed kernel
can be given as a *reference function* written with a typed frontend: a
Triton kernel verified against a `stile.torch` reference, a Pallas kernel
against a `stile.jax` reference. The reference's `ExprType` is compared
against the kernel's by the same machinery the spec string uses, and its
ShapeType + dtype are checked against the declared output.

Pallas tests run under `interpret=True` (CPU). Triton verification is
static (decoration time) and runs without a GPU; the full launch is
gated on CUDA.
"""
import pytest

from stile import dim

try:
    import jax
    import jax.numpy as jnp
    import jax.experimental.pallas as pl
    import stile.jax as tjax
    import stile.jax.pallas as tpl
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

import stile.triton as ttl
try:
    import torch
    import stile.torch as tt
    from stile.torch._core import TypedTorchTensor
    from stile.type import Type, Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if ttl.HAS_TRITON:
    import triton.language as tl
else:
    tl = None

REQUIRES_JAX = pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
REQUIRES_TORCH = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
REQUIRES_TRITON = pytest.mark.skipif(
    not ttl.HAS_TRITON, reason="Triton not installed",
)
try:
    REQUIRES_CUDA = pytest.mark.skipif(
        not (HAS_TORCH and torch.cuda.is_available()),
        reason="needs torch + CUDA (run on spark)",
    )
except Exception:
    REQUIRES_CUDA = pytest.mark.skipif(True, reason="needs torch + CUDA")


# --- Pallas vs jax reference -------------------------------------------

@REQUIRES_JAX
def test_pallas_reference_scalar_multiply(reset):
    """
    The smallest reference: `lambda x: x * 2` stands in for the spec
    string `"2 * TPN"`. Verifier accepts; runtime matches `2 * x`.
    """
    N = dim("XFN", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def kernel(x_ref : tpl.TypedRef, o_ref : tpl.TypedOutputRef):
        o_ref.assign(x_ref.load() * 2)

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(lambda x: x * 2, x.type.st, jnp.float32),
    )(x)

    assert isinstance(result, tjax.TypedJaxArray)
    assert jnp.allclose(result.arr, x.arr * 2)


@REQUIRES_JAX
def test_pallas_reference_rejects_wrong_value(reset):
    """
    Kernel doubles but the reference says triple — the ExprTypes
    don't normalize equal, so `.assign` rejects before the store.
    """
    N = dim("XFN2", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def kernel(x_ref, o_ref):
        o_ref.assign(x_ref.load() * 2)

    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(lambda x: x * 3, x.type.st, jnp.float32),
    )
    with pytest.raises(ValueError, match="does not match spec"):
        runner(x)


@REQUIRES_JAX
def test_pallas_reference_rejects_wrong_dtype(reset):
    """
    The reference computes in f32 (inputs are f32) but the output is
    declared bf16 — the dtype check rejects at resolution time.
    """
    N = dim("XFN3", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def kernel(x_ref, o_ref):
        o_ref.assign(x_ref.load() * 2)

    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(lambda x: x * 2, x.type.st, jnp.bfloat16),
    )
    with pytest.raises(ValueError, match="dtype"):
        runner(x)


@REQUIRES_JAX
def test_pallas_reference_rejects_wrong_shape(reset):
    """
    The reference reduces the dim away (shape `()`), disagreeing with
    the declared output shape — the ShapeType check rejects.
    """
    N = dim("XFN4", 8)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def kernel(x_ref, o_ref):
        o_ref.assign(x_ref.load() * 2)

    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(lambda x: x.sum(N), x.type.st, jnp.float32),
    )
    with pytest.raises(ValueError, match="shape"):
        runner(x)


@REQUIRES_JAX
def test_pallas_reference_two_inputs_positional(reset):
    """
    Two inputs: the reference receives them positionally in call
    order. `lambda a, b: a + b` stands in for `"a + b"`.
    """
    N = dim("XFADD", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), N, name="a")
    b = tjax.random.normal(jax.random.PRNGKey(1), N, name="b")

    def kernel(a_ref, b_ref, o_ref):
        o_ref.assign(a_ref.load() + b_ref.load())

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(lambda a, b: a + b, a.type.st, jnp.float32),
    )(a, b)

    assert jnp.allclose(result.arr, a.arr + b.arr, atol=1e-5)


@REQUIRES_JAX
def test_pallas_reference_matmul_tiled(reset):
    """
    Tiled matmul against a jax reference. The reference runs on the
    full inputs to produce the full-output ExprType, which is then
    tile-restricted exactly like a parsed spec — so the per-block
    `assign(...)` certifies the tile equals the reference's matmul
    restricted to that tile.
    """
    M, N, K = dim("XFM", 16), dim("XFN5", 8), dim("XFK", 16)
    BM, BK = 8, 8
    a = tjax.random.normal(jax.random.PRNGKey(0), M, N, name="a")
    b = tjax.random.normal(jax.random.PRNGKey(1), N, K, name="b")

    def kernel(a_ref, b_ref, o_ref):
        o_ref.assign(tjax.einsum(
            a_ref.load(), b_ref.load(), "XFM XFN5, XFN5 XFK -> XFM XFK",
        ))

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec(
            lambda a, b: tjax.einsum(a, b, "XFM XFN5, XFN5 XFK -> XFM XFK"),
            (M, K), jnp.float32,
        ),
        grid=(M.size // BM, K.size // BK),
        in_specs=[
            pl.BlockSpec((BM, N.size), lambda m, k: (m, 0)),
            pl.BlockSpec((N.size, BK), lambda m, k: (0, k)),
        ],
        out_specs=pl.BlockSpec((BM, BK), lambda m, k: (m, k)),
    )(a, b)

    assert jnp.allclose(result.arr, a.arr @ b.arr, atol=1e-5)


# --- Triton vs torch reference -----------------------------------------

@REQUIRES_TRITON
@REQUIRES_TORCH
def test_triton_reference_verification(reset):
    """
    Verification half only (no GPU): the `@ttl.jit` decorator runs the
    torch reference `lambda X: X * 2` on symbolic inputs to get the
    expected ExprType, then checks the kernel body against it.
    """
    N = dim("XFTTN", 128)
    BLOCK = 32

    @ttl.jit(
        spec=lambda X: X * 2,
        inputs={"X_ptr" : "X:XFTTN"},
        out_shape=(N,),
        out_dtype=torch.float32,
    )
    def double(X_ptr, o_ptr, BLOCK : tl.constexpr):
        pid = tl.program_id(0)
        X = ttl.load(X_ptr, N[pid * BLOCK : (pid + 1) * BLOCK])
        result = X * 2
        ttl.store(o_ptr, result, N[pid * BLOCK : (pid + 1) * BLOCK])

    assert isinstance(double, ttl.TypedTritonKernel)


@REQUIRES_TRITON
@REQUIRES_TORCH
def test_triton_reference_rejects_mismatch(reset):
    """
    Kernel stores `X * 3` but the reference says `X * 2` — decoration
    raises when the stored ExprType doesn't match the reference's.
    """
    N = dim("XFTTN2", 128)
    BLOCK = 32

    with pytest.raises(AssertionError, match="does not match spec"):
        @ttl.jit(
            spec=lambda X: X * 2,
            inputs={"X_ptr" : "X:XFTTN2"},
            out_shape=(N,),
            out_dtype=torch.float32,
        )
        def bad(X_ptr, o_ptr, BLOCK : tl.constexpr):
            pid = tl.program_id(0)
            X = ttl.load(X_ptr, N[pid * BLOCK : (pid + 1) * BLOCK])
            ttl.store(o_ptr, X * 3, N[pid * BLOCK : (pid + 1) * BLOCK])


@REQUIRES_TRITON
@REQUIRES_TORCH
def test_triton_reference_rejects_wrong_shape(reset):
    """
    The reference's output shape (a reduction to a scalar over the
    block dim) disagrees with the declared `out_shape` — decoration
    raises from the ShapeType check before touching the kernel body.
    """
    N = dim("XFTTN3", 128)
    BLOCK = 32

    with pytest.raises(ValueError, match="shape"):
        @ttl.jit(
            spec=lambda X: X.sum(N),
            inputs={"X_ptr" : "X:XFTTN3"},
            out_shape=(N,),
            out_dtype=torch.float32,
        )
        def bad(X_ptr, o_ptr, BLOCK : tl.constexpr):
            pid = tl.program_id(0)
            X = ttl.load(X_ptr, N[pid * BLOCK : (pid + 1) * BLOCK])
            ttl.store(o_ptr, X * 2, N[pid * BLOCK : (pid + 1) * BLOCK])


@REQUIRES_CUDA
@REQUIRES_TRITON
def test_triton_reference_scalar_multiply_launch(reset):
    """
    Full path on GPU: verify against the torch reference, emit, launch,
    and numerically check `x * 2`.
    """
    N = dim("XFTTN4", 128)
    BLOCK = 32

    @ttl.jit(
        spec=lambda X: X * 2,
        inputs={"X_ptr" : "X:XFTTN4"},
        out_shape=(N,),
        out_dtype=torch.float32,
    )
    def double(X_ptr, o_ptr, BLOCK : tl.constexpr):
        pid = tl.program_id(0)
        X = ttl.load(X_ptr, N[pid * BLOCK : (pid + 1) * BLOCK])
        result = X * 2
        ttl.store(o_ptr, result, N[pid * BLOCK : (pid + 1) * BLOCK])

    x_arr = torch.randn(N.size, device="cuda")
    x = TypedTorchTensor(x_arr, Type(st=(N,), et=Tensor(dims=(N,), name="X")))
    result = double[(N.size // BLOCK,)](x, BLOCK=BLOCK)
    assert torch.allclose(result.tensor, x_arr * 2, atol=1e-5)


@REQUIRES_CUDA
@REQUIRES_TRITON
def test_triton_reference_matmul_launch(reset):
    """
    Two-input matmul against a torch reference, full GPU launch. The
    reference `lambda A, B: tt.einsum(A, B, "M K, K N -> M N")` stands in
    for the spec `"(A:M K, B:K N -> M N)"` and receives the inputs
    positionally in `inputs={...}` order.
    """
    M = dim("XFMM", 32)
    N = dim("XFNN", 32)
    K = dim("XFKK", 32)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = K.size

    @ttl.jit(
        spec=lambda A, B: tt.einsum(A, B, "XFMM XFKK, XFKK XFNN -> XFMM XFNN"),
        inputs={"A_ptr" : "A:XFMM XFKK", "B_ptr" : "B:XFKK XFNN"},
        out_shape=(M, N),
        out_dtype=torch.float32,
    )
    def matmul(
        A_ptr, B_ptr, C_ptr,
        BLOCK_M : tl.constexpr, BLOCK_N : tl.constexpr, BLOCK_K : tl.constexpr,
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
    a = TypedTorchTensor(a_arr, Type(st=(M, K), et=Tensor(dims=(M, K), name="A")))
    b = TypedTorchTensor(b_arr, Type(st=(K, N), et=Tensor(dims=(K, N), name="B")))
    result = matmul[(M.size // BLOCK_M, N.size // BLOCK_N)](
        a, b, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    assert torch.allclose(result.tensor, a_arr @ b_arr, atol=2e-2, rtol=1e-3)
