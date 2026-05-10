"""
GPU-only Pallas tests. Skipped when no CUDA device is available
(e.g., on a Mac dev box). On a real CUDA host these run the same
typed kernels as `test_typed_pallas.py` but with `interpret=False`,
exercising the actual Pallas → Mosaic-GPU lowering and running the
kernel on the GPU.

What's covered today, and what's blocked upstream:

  - scalar_multiply, vector_add: pure-elementwise. Run on GPU. ✓
  - matmul, flash_attention: need either `dot_general` (Mosaic-GPU's
    `Lane` lowering doesn't yet support it) or axis reductions
    (also unimplemented in `Lane`). Tracked as upstream JAX issues;
    once they land, these will light up with no stile-side changes.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax
import stile.jax.pallas as tpl
from stile import dim, reset_stile


def _has_cuda():
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cuda(),
    reason="GPU-only tests; no CUDA device available.",
)


@pytest.fixture
def reset():
    yield
    reset_stile()


@pytest.fixture(scope="module")
def gpu_compiler_params():
    """Mosaic-GPU CompilerParams — required for `interpret=False` on a
    CUDA Pallas backend. Imported lazily inside the fixture so the
    import doesn't fail on Pallas-less hosts at collection time."""
    from jax.experimental.pallas import mosaic_gpu as plgpu
    return plgpu.CompilerParams()


def test_scalar_multiply_gpu(reset, gpu_compiler_params):
    """`x * 2` on real CUDA via Mosaic-GPU. Verifier proves the kernel
    matches `2 * TPN`; runtime executes the lowering on GPU."""
    # Mosaic-GPU's warpgroup transfer requires multiples of 128 bytes
    # (32 float32s); use 128 elements as the smallest comfortable size.
    N = dim("TPN", 128)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def kernel(x_ref, o_ref):
        o_ref.assign(x_ref.load() * 2)

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * TPN", x.type.st, jnp.float32),
        interpret=False,
        compiler_params=gpu_compiler_params,
    )(x)

    assert result.arr.device.platform == "gpu"
    assert jnp.allclose(result.arr, x.arr * 2)


def test_vector_add_gpu(reset, gpu_compiler_params):
    """`a + b` on real CUDA, two distinct labeled inputs. Verifier
    proves the kernel matches `a:VAN + b:VAN`."""
    N = dim("VAN", 128)
    a = tjax.random.normal(jax.random.PRNGKey(0), N, name="a")
    b = tjax.random.normal(jax.random.PRNGKey(1), N, name="b")

    def kernel(a_ref, b_ref, o_ref):
        o_ref.assign(a_ref.load() + b_ref.load())

    result = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("a:VAN + b:VAN", a.type.st, jnp.float32),
        interpret=False,
        compiler_params=gpu_compiler_params,
    )(a, b)

    assert result.arr.device.platform == "gpu"
    assert jnp.allclose(result.arr, a.arr + b.arr)


def test_scalar_multiply_wrong_factor_rejected_gpu(reset, gpu_compiler_params):
    """Verifier rejects the `*3` kernel for the `2 * TPN` spec — the
    rejection is at trace time, before lowering, so Pallas-GPU never
    even runs. Just confirms soundness machinery doesn't change with
    the GPU backend."""
    N = dim("TPN", 128)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def kernel(x_ref, o_ref):
        o_ref.assign(x_ref.load() * 3)

    runner = tpl.typed_pallas_call(
        kernel,
        out_type=tpl.OutputSpec("2 * TPN", x.type.st, jnp.float32),
        interpret=False,
        compiler_params=gpu_compiler_params,
    )
    with pytest.raises(ValueError, match="does not match spec"):
        runner(x)


# Upstream-blocked, kept here as documented xfails so they light up
# automatically once the Pallas Mosaic-GPU lowering grows the missing
# primitives. Each test references the specific lowering gap.

@pytest.mark.xfail(
    reason="Pallas-GPU's `dot_general` lowering isn't implemented for "
    "either `Lane` or `Warpgroup` lowering semantics. The supported "
    "GPU matmul path is the explicit `plgpu.wgmma` (Hopper) / "
    "`plgpu.tcgen05_mma` (Blackwell) intrinsics with appropriate "
    "tile/swizzle transforms — see `jax.experimental.pallas.ops.gpu."
    "blackwell_matmul_mgpu`. Wrapping that with stile typing is a real "
    "next step; this test pins down the trivial `tjax.einsum` path.",
)
def test_matmul_gpu(reset, gpu_compiler_params):
    M, N, K = dim("M", 128), dim("N", 128), dim("K", 128)
    BM, BK = 64, 64
    a = tjax.random.normal(jax.random.PRNGKey(0), M, N, name="a")
    b = tjax.random.normal(jax.random.PRNGKey(1), N, K, name="b")
    import jax.experimental.pallas as pl

    def kernel(a_ref, b_ref, o_ref):
        o_ref.assign(tjax.einsum(
            a_ref.load(), b_ref.load(), "M N, N K -> M K",
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
        interpret=False,
        compiler_params=gpu_compiler_params,
    )(a, b)
    assert jnp.allclose(result.arr, a.arr @ b.arr, atol=1e-3)
