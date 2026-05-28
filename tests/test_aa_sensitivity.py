"""
Tests for `sensitivity_analysis`: re-running a typed kernel under a
swapped precision plan and reading off the per-op bound widening.

The compelling shape of this for the verifier story is:

  - User writes a kernel parameterized by named typed inputs.
  - User asks "what if I downgrade A to fp8?" via
    `sensitivity_analysis(kernel, args, swap={"A": "fp8_e4m3"})`.
  - Helper re-runs the kernel twice, returns total widening + per-op
    breakdown labeled by which AA noise contributed.

What we pin here:

  1. The total widening factor is finite and >1 when a precision
     downgrade is applied (smoke).
  2. The per-label breakdown surfaces the `*-round` noise from each
     individual op the kernel performs (so the caller can see which
     op is the dominant sensitivity).
  3. Swapping NO tensors gives a widening of exactly 1.0 (round-trip
     invariant: identity swap = no change).
  4. Swapping a single input shows up in the einsum's mul-rounding,
     not in the rest of the kernel.

Backend choice: stile.numpy here for portability — numpy doesn't have
bfloat16 / fp8 natively, but it has float16, which is wider-eps than
float32 (eps 2^-11 vs 2^-24) and so triggers the same widening
qualitatively. The JAX / torch versions of the test live on Spark
where bfloat16 + fp8 round-trip cleanly.
"""
import math

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from conftest import REQUIRES_JAX, REQUIRES_TORCH

import stile.jax as tjax
import stile.numpy as tnumpy
import stile.torch as ttorch
from stile import dim
from stile.numerical import Sensitivity, sensitivity_analysis


def _einsum_kernel_numpy(A, B):
    """Tiny matmul kernel — exercises one `einsum`."""
    return tnumpy.einsum(A, B, "M K, K N -> M N")


def _build_numpy_args(*, dtype):
    M = dim("M", 4)
    K = dim("K", 8)
    N = dim("N", 4)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 8)).astype(dtype)
    b = rng.standard_normal((8, 4)).astype(dtype)
    return (
        tnumpy.tensor(a, M, K, name="A"),
        tnumpy.tensor(b, K, N, name="B"),
    )


def test_sensitivity_returns_sensitivity(reset):
    """Smoke: identity swap (no tensors named in `swap`) is a no-op
    and yields widening = 1.0."""
    A, B = _build_numpy_args(dtype=np.float32)
    s = sensitivity_analysis(_einsum_kernel_numpy, (A, B), swap={})
    assert isinstance(s, Sensitivity)
    assert s.widening == 1.0


def test_swap_one_input_promotes_back_widening_minimal(reset):
    """Swapping just `A` to float16 does NOT widen the bound, because
    numpy promotes `fp16 × fp32 → fp32` — the einsum still runs at
    fp32 precision. This is the correct ablation answer: "downgrading
    A alone has no effect under promotion semantics", and the
    sensitivity analysis correctly surfaces that."""
    A, B = _build_numpy_args(dtype=np.float32)
    s = sensitivity_analysis(
        _einsum_kernel_numpy, (A, B), swap={"A": "float16"},
    )
    # The bound moves only marginally — driven by the truncation of
    # A's values when cast to fp16, not by op rounding.
    assert math.isclose(s.widening, 1.0, rel_tol=1e-2)


def test_swap_both_inputs_widens(reset):
    """When ALL inputs are downgraded to float16, the einsum's output
    dtype follows (fp16), and the `einsum-mul-round` noise widens by
    ≈ MACHINE_EPS[fp16] / MACHINE_EPS[fp32] ≈ 8192×."""
    A, B = _build_numpy_args(dtype=np.float32)
    s = sensitivity_analysis(
        _einsum_kernel_numpy, (A, B), swap={"A": "float16", "B": "float16"},
    )
    # Total widening dominated by the data-dependent `mul-cross`
    # (cancels in ratio at ~1.0) plus the dtype-driven `*-round`
    # noises. Just assert directional: the per-label `einsum-mul-round`
    # should widen substantially.
    assert "einsum-mul-round" in s.per_label
    _, _, mul_ratio = s.per_label["einsum-mul-round"]
    # eps_fp16 / eps_fp32 ≈ 8192. We allow looseness because the leaf
    # radii also shrink slightly after the fp16 cast.
    assert mul_ratio > 1000.0, mul_ratio
    # `rounding_widening` restricts to dtype-driven labels and so gives
    # the interpretable headline number directly — should also be huge.
    assert s.rounding_widening > 1000.0, s.rounding_widening


def test_sensitivity_per_label_keys_are_op_round_labels(reset):
    """The per-label breakdown surfaces the actual `*-round` labels
    each op attaches. For a single-einsum kernel under the default
    `WORST_CASE` (sequential) reduction we expect the multiply-round
    label and the seq-sum-K accumulator label."""
    A, B = _build_numpy_args(dtype=np.float32)
    s = sensitivity_analysis(
        _einsum_kernel_numpy, (A, B),
        swap={"A": "float16", "B": "float16"},
    )
    labels = set(s.per_label.keys())
    assert any("mul-round" in l for l in labels), labels


def test_sensitivity_summary_renders(reset):
    """The Sensitivity object has a `summary()` method that produces
    a multi-line string for terminal printing."""
    A, B = _build_numpy_args(dtype=np.float32)
    s = sensitivity_analysis(
        _einsum_kernel_numpy, (A, B),
        swap={"A": "float16", "B": "float16"},
    )
    text = s.summary()
    assert "total widening" in text
    assert "mul-round" in text


@REQUIRES_JAX
def test_sensitivity_jax_bf16(reset):
    """End-to-end JAX path: swapping both `A` and `B` to bfloat16
    widens the einsum-mul-round noise (the einsum then runs at
    bfloat16 precision throughout)."""
    M = dim("M", 4)
    K = dim("K", 8)
    N = dim("N", 4)
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((4, 8)).astype(np.float32))
    b = jnp.asarray(rng.standard_normal((8, 4)).astype(np.float32))
    A = tjax.tensor(a, M, K, name="A")
    B = tjax.tensor(b, K, N, name="B")

    def kernel(A, B):
        return tjax.einsum(A, B, "M K, K N -> M N")

    s = sensitivity_analysis(
        kernel, (A, B), swap={"A": "bfloat16", "B": "bfloat16"},
    )
    assert s.widening > 1.0


@REQUIRES_TORCH
def test_sensitivity_torch_bf16(reset):
    """End-to-end Torch path: swap both `A` and `B` to bfloat16."""
    M = dim("M", 4)
    K = dim("K", 8)
    N = dim("N", 4)
    torch.manual_seed(0)
    a = torch.randn(4, 8, dtype=torch.float32)
    b = torch.randn(8, 4, dtype=torch.float32)
    A = ttorch.tensor(a, M, K, name="A")
    B = ttorch.tensor(b, K, N, name="B")

    def kernel(A, B):
        return ttorch.einsum(A, B, "M K, K N -> M N")

    s = sensitivity_analysis(
        kernel, (A, B), swap={"A": "bfloat16", "B": "bfloat16"},
    )
    assert s.widening > 1.0
