"""
End-to-end demo: AA-predicted rounding-error bound on a TF32 matmul,
compared against the actual measured error from running on Spark's
tensor cores.

For each `(M, N, K)` matmul shape, we:

  1. Build the symbolic matmul ET:
        c[i, j] = Σ_k A[i, k] · B[k, j]
     using stile's raw `Reduce(sum, K, Mul(A, B))` structure.

  2. Build `AffineForm` leaves for `A` and `B` whose ranges match the
     actual fp32 input ranges (computed from `torch.randn`-sampled
     tensors).

  3. Evaluate the ET under a `HardwareNumerics` that models TF32
     tensor cores: TF32 epsilon (`2^-11`) on the multiplications,
     FP32 epsilon (`2^-24`) on the accumulator's tree reduction.

  4. Use `rounding_error_bound` to extract the per-cell error
     attributable solely to finite-precision rounding (separates
     from input-range uncertainty).

  5. Run the actual matmul on Spark with TF32 enabled; compute the
     measured per-cell error against an FP64 reference.

  6. Assert that the AA-predicted bound dominates the measured
     error — i.e. the bound is a valid (conservative) upper bound.

This is the substrate for the fp8-quantization-tracing story: once
we know AA gives valid bounds for a given hardware model, the same
machinery lets us pose "is this kernel safe to downgrade to fp8?"
as a comparison `rounding_error_bound(et, hw=fp8) ≤ accuracy_budget`.
"""
import pytest

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_TORCH = False
    HAS_CUDA = False

from stile import dim
from stile.type import Tensor, BinaryOp, Reduce
from stile.numerical import (
    AffineForm, fresh_noise, evaluate, rounding_error_bound,
    HardwareNumerics,
)


_REQUIRES_CUDA = pytest.mark.skipif(
    not (HAS_TORCH and HAS_CUDA),
    reason="needs torch + CUDA (run on spark)",
)


# Hardware model for NVIDIA TF32 matmul: TF32 inputs (10-bit mantissa,
# ε = 2^-11), FP32 accumulator (ε = 2^-24), tree reduction approximates
# the MMA's hardware-fused per-tile semantics conservatively.
NVIDIA_TF32_MATMUL = HardwareNumerics(
    name="nvidia_tf32_matmul",
    default_dtype="tf32",
    accumulator_dtype="float32",
    reduction_order="tree",
)


@_REQUIRES_CUDA
@pytest.mark.parametrize("m,n,k", [
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 64),
])
def test_tf32_matmul_aa_bound_dominates_measured(reset, m, n, k):
    torch.manual_seed(0)
    M = dim(f"DemoM_{m}_{n}_{k}", m)
    N = dim(f"DemoN_{m}_{n}_{k}", n)
    K = dim(f"DemoK_{m}_{n}_{k}", k)

    # FP64 reference + FP32 test inputs.
    a64 = torch.randn(m, k, device="cuda", dtype=torch.float64)
    b64 = torch.randn(k, n, device="cuda", dtype=torch.float64)
    a_input_radius = float(a64.abs().max().item())
    b_input_radius = float(b64.abs().max().item())

    a32 = a64.to(torch.float32)
    b32 = b64.to(torch.float32)

    # --- AA prediction ---
    a_form = AffineForm.with_noise(0.0, fresh_noise("A"), a_input_radius)
    b_form = AffineForm.with_noise(0.0, fresh_noise("B"), b_input_radius)
    leaves = {"A": a_form, "B": b_form}

    et = Reduce(
        op="sum", dim=K,
        child=BinaryOp(
            op="*",
            lhs=Tensor(dims=(M, K), name="A"),
            rhs=Tensor(dims=(K, N), name="B"),
        ),
    )
    result = evaluate(et, leaves, hardware=NVIDIA_TF32_MATMUL)
    predicted_error = rounding_error_bound(result, leaves)

    # --- Measured error: TF32 matmul vs FP64 reference ---
    prev_tf32_allow = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        c_tf32 = a32 @ b32   # actually uses TF32 tensor cores on sm_80+
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_allow
    c_ref64 = a64 @ b64

    measured_error = (
        (c_tf32.to(torch.float64) - c_ref64).abs().max().item()
    )

    # The AA bound is conservative — it must be ≥ the actual worst-cell
    # error. (If this assertion fires the bound is unsound for this
    # hardware model, which is a much bigger deal than the bound being
    # loose.)
    assert measured_error <= predicted_error, (
        f"AA bound violated! shape=({m},{n},{k}): "
        f"predicted={predicted_error:.4e}, measured={measured_error:.4e}"
    )

    # Diagnostic — typically the bound is loose by ~2-10× for tree
    # reductions vs hardware-fused MMA. Show the ratio.
    ratio = predicted_error / max(measured_error, 1e-30)
    print(
        f"\n  matmul {m}×{k} @ {k}×{n}: "
        f"predicted={predicted_error:.4e}, measured={measured_error:.4e}, "
        f"bound/actual ratio={ratio:.2f}×"
    )
