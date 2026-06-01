"""
Hardware models for the numerical evaluator.

Different backends have wildly different numerical behavior on the
same kernel. A `tl.dot` on NVIDIA tensor cores upcasts TF32 inputs to
FP32 internally and accumulates in FP32 with hardware-fused
reduction. The same op on Trainium's Tensor Engine upcasts everything
to FP32 with an unknown reduction order over a 128-element native
tile. On a CPU backend with naive scalar codegen, it's sequential FP32
all the way.

A `HardwareNumerics` lets the caller describe what their target backend
actually does, so the AA evaluator can attach the right rounding noise
at the right places. The default (`WORST_CASE`) is the most pessimistic
shape — sequential reduction at the input dtype with no accumulator
upcast — so it's a valid upper bound for any real backend. Subclasses
should refine when their actual rounding is tighter.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Literal

from .affine import MACHINE_EPS


ReductionOrder = Literal["sequential", "tree", "hw_fused"]


@dataclass(frozen=True)
class HardwareNumerics:
    """
    Backend-specific numerical model the evaluator reads when
    attaching rounding noise.

    Fields:

    - `default_dtype`: dtype of elementwise scalar ops (add, sub, mul,
      div, exp, …). Single-op rounding error is bounded by
      `MACHINE_EPS[default_dtype] · |value|`.

    - `accumulator_dtype`: dtype used inside reductions and matmul
      accumulators. Often higher precision than `default_dtype` (e.g.
      `bfloat16` inputs, `float32` accumulator). When `None`, falls
      back to `default_dtype`.

    - `reduction_order`: how an N-element reduction is broken down.
        * `"sequential"` (worst case): `s = x_0; for i: s = s + x_i`.
          `N - 1` rounding ops, partial sums grow linearly.
        * `"tree"`: pairwise tree reduction. `N - 1` rounding ops
          total, but partial sums stay bounded; error scales with
          `log₂ N · ε`.
        * `"hw_fused"`: hardware does the full N-wide reduction with
          one rounding step (or `ceil(N / native_tile_size)` if
          `native_tile_size < N`).

    - `native_tile_size`: hardware-fused reduction width. A reduction
      of `N` elements is chunked into `ceil(N / native_tile_size)`
      tiles, each tile's internal reduction follows `reduction_order`,
      and the per-tile partial sums then combine.

      For `reduction_order == "hw_fused"`, `native_tile_size` is the
      width of each fused step. For `"sequential"` / `"tree"`, it
      controls when reductions cross the natural-tile boundary (the
      tile-internal accumulator may be wider than the cross-tile one,
      but we don't model that distinction yet — future work).

    - `matmul_input_dtype` / `matmul_accumulator_dtype`: per-op
      override for matrix multiplications, which often run with
      input upcast and FP32 accumulation. Defaults to `default_dtype`
      / `accumulator_dtype`. Used only by the matmul handler (when
      it lands — not in the prototype).
    """
    name : str = "worst_case"
    default_dtype : str = "float32"
    accumulator_dtype : "str | None" = None
    reduction_order : ReductionOrder = "sequential"
    native_tile_size : int = 1
    matmul_input_dtype : "str | None" = None
    matmul_accumulator_dtype : "str | None" = None

    def __post_init__(self):
        if self.default_dtype not in MACHINE_EPS:
            raise ValueError(
                f"unknown default_dtype {self.default_dtype!r}; "
                f"expected one of {sorted(MACHINE_EPS)}"
            )
        if (self.accumulator_dtype is not None
                and self.accumulator_dtype not in MACHINE_EPS):
            raise ValueError(
                f"unknown accumulator_dtype "
                f"{self.accumulator_dtype!r}"
            )

    @property
    def acc_dtype(self) -> str:
        """`accumulator_dtype` with `default_dtype` fallback."""
        return self.accumulator_dtype or self.default_dtype

    @property
    def acc_eps(self) -> float:
        """Machine epsilon for the accumulator."""
        return MACHINE_EPS[self.acc_dtype]

    @property
    def default_eps(self) -> float:
        return MACHINE_EPS[self.default_dtype]


# --- Preset models --------------------------------------------------

# Most pessimistic — sequential, no upcast, no fused MMA.
# Every other real backend should give a tighter or equal bound.
WORST_CASE = HardwareNumerics(
    name="worst_case",
    default_dtype="float32",
    reduction_order="sequential",
)

# NVIDIA tensor cores on Hopper / Ampere doing TF32 input, FP32 accum,
# fused 8-wide MMA reductions. The scalar ops outside the MMA are
# whatever the user wrote (default fp32 here).
NVIDIA_TENSOR_CORE_TF32 = HardwareNumerics(
    name="nvidia_tc_tf32",
    default_dtype="float32",
    accumulator_dtype="float32",
    reduction_order="hw_fused",
    native_tile_size=8,
    matmul_input_dtype="tf32",
    matmul_accumulator_dtype="float32",
)

# Trainium's Tensor Engine — everything upcasts to FP32 internally,
# 128-wide native tile, reduction order is unspecified (treat as
# sequential at the accumulator dtype = worst-case).
TRAINIUM_TENSOR_ENGINE = HardwareNumerics(
    name="trainium_te",
    default_dtype="bfloat16",
    accumulator_dtype="float32",
    reduction_order="sequential",   # actual order is unknown / opaque
    native_tile_size=128,
    matmul_input_dtype="bfloat16",
    matmul_accumulator_dtype="float32",
)

# TPU MXU — bf16 input, fp32 accumulate, fused 128-wide MMA.
TPU_MXU = HardwareNumerics(
    name="tpu_mxu",
    default_dtype="bfloat16",
    accumulator_dtype="float32",
    reduction_order="hw_fused",
    native_tile_size=128,
    matmul_input_dtype="bfloat16",
    matmul_accumulator_dtype="float32",
)


# --- Active-hardware stack -----------------------------------------
# A stack of `HardwareNumerics` instances pushed by `numerical_context`.
# Top of stack wins; empty stack means `WORST_CASE` is implied.
_active_hardware_stack : "list[HardwareNumerics]" = []


def active_hardware() -> HardwareNumerics:
    """
    The current `HardwareNumerics` in effect. Reads the top of the
    `numerical_context` stack; falls back to `WORST_CASE` when no
    context is active. Every typed op handler that wants to attach
    eager AA noise reads this — backends ship their own preset and
    push it via `numerical_context(hardware=NVIDIA_TENSOR_CORE_TF32)`
    or similar.
    """
    if _active_hardware_stack:
        return _active_hardware_stack[-1]
    return WORST_CASE


@contextlib.contextmanager
def numerical_context(*, hardware : HardwareNumerics):
    """
    Push `hardware` onto the active-numerics stack for the duration of
    the `with` block. Nested contexts shadow outer ones; on exit the
    previous model is restored.

    Typical use is via `stile.verified(hardware=...)`, which bundles
    this with `stile.scope()`; a bare `numerical_context(...)` is
    useful when the caller manages dim scope separately.
    """
    _active_hardware_stack.append(hardware)
    try:
        yield hardware
    finally:
        popped = _active_hardware_stack.pop()
        assert popped is hardware
