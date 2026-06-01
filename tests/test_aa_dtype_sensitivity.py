"""
Per-tensor dtype drives the AA rounding noise on pointwise ops.

Background: pointwise op handlers (`+`, `*`, `exp`, `sin`, …) used
to consult `active_hardware().default_eps` for their rounding noise.
After the dtype-driven refactor, they read the *output* dtype off
the result array and look up `MACHINE_EPS[dtype]` directly. That
matches how mixed-precision programs actually work: each tensor has
its own dtype, the result dtype follows the library's promotion
rules, and the AA bound reflects whatever precision the result
actually lives at.

We pin a few behaviors here:

  1. Same kernel, fp32 vs bfloat16 inputs → bfloat16's wider epsilon
     produces a wider AA rounding bound *with no hardware context*.
  2. `dtype_name_of` correctly maps dtype objects from each backend
     to MACHINE_EPS keys.
  3. The fp32 bound is unchanged by the migration (regression check).
"""
import math

import jax.numpy as jnp
import numpy as np
import torch

from conftest import REQUIRES_JAX, REQUIRES_TORCH

from stile import dim
from stile.numerical import (
    MACHINE_EPS, dtype_name_of, rounding_error_bound,
)
from stile.numpy._core import TypedNumpyArray
from stile.torch._core import TypedTorchTensor
from stile.type import Tensor, Type


def test_dtype_name_of_resolves_numpy_dtypes():
    """numpy float32 / float64 / bool resolve to MACHINE_EPS keys."""
    assert dtype_name_of(np.zeros(1, dtype=np.float32)) == "float32"
    assert dtype_name_of(np.zeros(1, dtype=np.float64)) == "float64"
    # Unmodeled dtype (bool) → None (no rounding noise attached).
    assert dtype_name_of(np.zeros(1, dtype=np.bool_)) is None


def test_dtype_name_of_handles_none():
    """Symbolic-only typed values (arr is None) → None dtype."""
    assert dtype_name_of(None) is None


@REQUIRES_JAX
def test_dtype_name_of_jax_dtypes():
    """JAX float32 / bfloat16 resolve to MACHINE_EPS keys."""
    assert dtype_name_of(jnp.zeros(1, dtype=jnp.float32)) == "float32"
    assert dtype_name_of(jnp.zeros(1, dtype=jnp.bfloat16)) == "bfloat16"


@REQUIRES_TORCH
def test_dtype_name_of_torch_dtypes():
    """PyTorch float32 / bfloat16 / float16 resolve to MACHINE_EPS keys."""
    assert dtype_name_of(torch.zeros(1, dtype=torch.float32)) == "float32"
    assert dtype_name_of(torch.zeros(1, dtype=torch.bfloat16)) == "bfloat16"
    assert dtype_name_of(torch.zeros(1, dtype=torch.float16)) == "float16"


def test_pointwise_add_eps_follows_output_dtype(reset):
    """
    `a + b` at fp32 attaches an `add-round` noise of magnitude
    `MACHINE_EPS["float32"] · |a+b|_max`. The eps lookup is now
    per-op, not per-context, so two TypedNumpyArrays of different
    dtypes give different rounding contributions to the result.
    """
    N = dim("DtypeN", 4)
    typ = Type(st=(N,), et=Tensor(dims=(N,), name="X"))
    a32 = TypedNumpyArray(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), typ)
    b32 = TypedNumpyArray(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), typ)
    c32 = a32 + b32

    # The non-leaf noise mass on c32.aa = rounding contribution.
    leaves = {"a32" : a32.aa, "b32" : b32.aa}
    rounding_fp32 = rounding_error_bound(c32.aa, leaves)
    # |a+b| is constant 1.5, so the rounding noise = 1.5 · eps_fp32.
    assert math.isclose(
        rounding_fp32, 1.5 * MACHINE_EPS["float32"], rel_tol=1e-6,
    )


@REQUIRES_TORCH
def test_pointwise_add_widens_for_bfloat16(reset):
    """
    Same `a + b`, but bfloat16 inputs → bfloat16 result → bfloat16
    eps. No hardware context manager involved: the dtype of the
    output tensor drives the rounding noise.
    """
    N = dim("DtypeBfN", 4)
    typ = Type(st=(N,), et=Tensor(dims=(N,), name="X"))
    a_bf = TypedTorchTensor(
        torch.ones(4, dtype=torch.bfloat16), typ,
    )
    b_bf = TypedTorchTensor(
        0.5 * torch.ones(4, dtype=torch.bfloat16), typ,
    )
    c_bf = a_bf + b_bf

    a_f32 = TypedTorchTensor(torch.ones(4, dtype=torch.float32), typ)
    b_f32 = TypedTorchTensor(
        0.5 * torch.ones(4, dtype=torch.float32), typ,
    )
    c_f32 = a_f32 + b_f32

    bound_bf = rounding_error_bound(
        c_bf.aa, {"a" : a_bf.aa, "b" : b_bf.aa},
    )
    bound_f32 = rounding_error_bound(
        c_f32.aa, {"a" : a_f32.aa, "b" : b_f32.aa},
    )

    # bfloat16 eps is 2^-8, fp32 eps is 2^-24 ⇒ the bf16 bound should
    # be ~2^16 ≈ 65,000× larger.
    ratio = bound_bf / bound_f32
    expected = MACHINE_EPS["bfloat16"] / MACHINE_EPS["float32"]
    assert math.isclose(ratio, expected, rel_tol=1e-3), (
        f"expected ratio ≈ {expected:.2e}, got {ratio:.2e}"
    )
