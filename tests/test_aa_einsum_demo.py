"""
End-to-end demo: eager affine-arithmetic bound on a tjax matmul,
validated against the measured fp32-vs-fp64 rounding error.

Unlike `test_numerical_demo_tf32.py` (which builds the matmul ET by
hand and walks it through the lazy ET evaluator), this exercises the
*eager* path: per-op AA composition wired into `tjax.einsum`. The user
calls `tjax.einsum(A, B, "i k, k j -> i j")` and the resulting typed
value's `.aa` already encodes the AA bound — no extra ET-evaluation
step required.

Test plan:

  1. **Default numerics (CPU_FP32)**: matmul of two fp32 tensors versus
     an fp64 reference. The AA bound is loose (worst-case tree
     reduction) but valid.

  2. **TF32 hardware model**: same matmul under a hardware model
     reflecting NVIDIA tensor-core TF32 (10-bit mantissa input,
     fp32 accumulator). On CPU we still compute the matmul in plain
     fp32 — the point of this branch is that the AA bound *widens*
     to reflect the TF32 model, while still soundly bounding the
     fp32 error.

The CUDA-only sibling `test_numerical_demo_tf32.py` runs the actual
TF32 tensor-core matmul on Spark and asserts the same kind of
inequality against measured hardware error. This file is CPU-only.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from conftest import REQUIRES_JAX

import stile.jax as tjax
from stile import dim
from stile.numerical import (
    HardwareNumerics, numerical_context, rounding_error_bound,
)


# --- Hardware models -------------------------------------------------
# CPU-side reference matmul: scalar accumulator at fp32 precision. The
# "tree" reduction order is the standard worst-case shape for matmul
# accumulators.
CPU_FP32 = HardwareNumerics(
    name="cpu_fp32",
    default_dtype="float32",
    accumulator_dtype="float32",
    reduction_order="tree",
)

# Tensor-core TF32: the MMA forces a TF32 multiplier (10-bit mantissa)
# even from fp32 operands; the accumulator stays fp32. The fp32 host-
# default applies to non-matmul ops; `matmul_input_dtype="tf32"` is
# what wires up the TF32 rounding noise inside the matmul handler.
NVIDIA_TF32_MATMUL = HardwareNumerics(
    name="nvidia_tf32_matmul",
    default_dtype="float32",
    accumulator_dtype="float32",
    reduction_order="tree",
    matmul_input_dtype="tf32",
    matmul_accumulator_dtype="float32",
)


@REQUIRES_JAX
@pytest.mark.parametrize("m,n,k", [
    (32, 32, 32),
    (64, 32, 64),
    (32, 64, 128),
])
def test_eager_aa_bounds_fp32_matmul(reset, m, n, k):
    """The eager `.aa` bound on `tjax.einsum(A, B, ...)` dominates the
    fp32-vs-fp64 measured error, under a `CPU_FP32` hardware model."""
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((m, k)).astype(np.float64)
    b_np = rng.standard_normal((k, n)).astype(np.float64)

    a32 = jnp.asarray(a_np, dtype=jnp.float32)
    b32 = jnp.asarray(b_np, dtype=jnp.float32)

    M = dim("M", m)
    N = dim("N", n)
    K = dim("K", k)
    A = tjax.tensor(a32, M, K, name="A")
    B = tjax.tensor(b32, K, N, name="B")

    # Snapshot the leaf AA forms *before* running the einsum, so the
    # rounding-error bound can later subtract them out (= consider only
    # the per-op rounding noise the einsum introduced).
    leaf_forms = {"A": A.aa, "B": B.aa}
    assert leaf_forms["A"] is not None and leaf_forms["B"] is not None

    with numerical_context(hardware=CPU_FP32):
        C = tjax.einsum(A, B, "M K, K N -> M N")

    assert C.aa is not None, "eager AA should propagate through einsum"

    # Per-cell rounding-error bound: AA noise mass that's *not* part of
    # the input leaves. (The leaf radius models input range, not
    # rounding error; we want the latter for the fp32-vs-fp64 check.)
    predicted_bound = rounding_error_bound(C.aa, leaf_forms)

    # Reference: fp64 matmul via numpy. JAX defaults to fp32 unless
    # `jax_enable_x64` is set process-wide, which would interfere with
    # the rest of the suite — numpy is simpler and just as exact.
    c_ref = (a_np @ b_np).astype(np.float64)
    c_fp32_np = np.asarray(C.arr).astype(np.float64)
    measured_error = float(np.abs(c_fp32_np - c_ref).max())

    assert measured_error <= predicted_bound, (
        f"AA bound violated! shape=({m},{n},{k}): "
        f"predicted={predicted_bound:.4e}, measured={measured_error:.4e}"
    )

    # Diagnostic — the CPU_FP32 bound is the worst-case tree-reduction
    # bound, so 10-100× looseness vs measured is normal.
    ratio = predicted_bound / max(measured_error, 1e-30)
    print(
        f"\n  tjax.einsum {m}×{k} @ {k}×{n} [CPU_FP32]: "
        f"predicted={predicted_bound:.4e}, "
        f"measured={measured_error:.4e}, ratio={ratio:.2f}×"
    )


@REQUIRES_JAX
def test_tf32_model_widens_aa_bound(reset):
    """Swapping the hardware model from CPU_FP32 to NVIDIA_TF32_MATMUL
    must widen the AA rounding bound, since TF32 multiplications carry
    a much larger per-op epsilon than fp32.

    NB: with wide-radius leaves the AA *multiplication cross-term* (a
    linearization noise, not a rounding noise) dominates the bound and
    swamps the dtype-epsilon difference. We pick narrow-radius inputs
    here so the dtype-rounding contribution is comparable to (and the
    one that actually changes between) the two hardware models.
    """
    rng = np.random.default_rng(1)
    m, n, k = 64, 64, 128
    # Inputs centered at 1.0 with radius ~0.03 → mul cross-term scale
    # is r_A·r_B·K ≈ 0.12 while the *-round noise scales like
    # |center_A|·|center_B|·default_eps·K. With centers ≈ 1, the
    # rounding term is ≈ K · default_eps. fp32 default_eps ≈ 1.19e-7
    # (so ≈ 1.5e-5, dwarfed by the cross-term); tf32 ≈ 9.77e-4 (so
    # ≈ 0.125, dominates). The two regimes differ by orders of
    # magnitude in the dtype-contribution and the test catches it.
    a32 = jnp.asarray(
        (1.0 + 0.01 * rng.standard_normal((m, k))).astype(np.float32),
    )
    b32 = jnp.asarray(
        (1.0 + 0.01 * rng.standard_normal((k, n))).astype(np.float32),
    )

    M = dim("M", m)
    N = dim("N", n)
    K = dim("K", k)
    A = tjax.tensor(a32, M, K, name="A")
    B = tjax.tensor(b32, K, N, name="B")
    leaf_forms = {"A": A.aa, "B": B.aa}

    with numerical_context(hardware=CPU_FP32):
        C_fp32 = tjax.einsum(A, B, "M K, K N -> M N")
    with numerical_context(hardware=NVIDIA_TF32_MATMUL):
        C_tf32 = tjax.einsum(A, B, "M K, K N -> M N")

    bound_fp32 = rounding_error_bound(C_fp32.aa, leaf_forms)
    bound_tf32 = rounding_error_bound(C_tf32.aa, leaf_forms)

    assert bound_tf32 > bound_fp32, (
        f"expected TF32 bound > fp32 bound; got "
        f"tf32={bound_tf32:.4e}, fp32={bound_fp32:.4e}"
    )
    print(
        f"\n  AA bound on {m}×{k}@{k}×{n} (narrow leaves): "
        f"fp32={bound_fp32:.4e}, tf32={bound_tf32:.4e}, "
        f"widening factor={bound_tf32/bound_fp32:.2f}×"
    )
