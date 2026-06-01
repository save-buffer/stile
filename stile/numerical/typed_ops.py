"""
Per-op AA composition helpers, shared by `stile.jax` / `stile.torch` /
`stile.numpy`'s typed op handlers.

The eager-AA story: each typed op (`exp`, `add`, `einsum`, …) takes
the input typed values' `.aa` fields plus per-op metadata (the output
dtype for pointwise ops; the active `HardwareNumerics` for matmul /
reduction ops) and produces the output's new `.aa`. These helpers
centralize the per-op affine arithmetic so each backend's op handler
is a one-liner.

Each helper accepts `AffineForm | None` and returns `AffineForm | None`,
propagating `None` when any input AA is missing (e.g., the typed value
was constructed with `aa=None` to opt out of AA tracking).

Design note — dtype source: pointwise ops read their rounding epsilon
off the *output* dtype, which the backend handler hands in directly
(typically `_dtype_name(new_arr)` after computing the result). Matmul
and reduction ops keep a `HardwareNumerics` argument because the mma /
hardware-fused-reduction semantics aren't recoverable from operand
dtypes alone — TF32 forces a TF32 multiplier on fp32 inputs, TPU MXU
upcasts bf16 internally, Trainium does fp32-accumulate regardless of
input dtype, etc.
"""
from __future__ import annotations

import math

from .affine import (
    AffineForm, add, sub, mul, div, exp as aa_exp, sqrt as aa_sqrt,
    maximum as aa_maximum, affine_unary, round_fp, MACHINE_EPS,
)
from .hardware import HardwareNumerics
from .evaluator import _sum_reduction
from ..type import dim_name, dim_size, as_int


# Map dtype-name strings that backends use to the canonical
# `MACHINE_EPS` keys. JAX / numpy report `dtype.name` as 'float32',
# 'bfloat16', etc. — already canonical. PyTorch matches. The FP8
# family has a couple of variant names across libraries.
_DTYPE_ALIASES = {
    "float8_e4m3fn" : "fp8_e4m3",
    "float8_e4m3" :   "fp8_e4m3",
    "float8_e5m2" :   "fp8_e5m2",
    "float8_e3m4" :   "fp8_e3m4",
}


def dtype_name_of(arr) -> "str | None":
    """
    Map an array's dtype to a `MACHINE_EPS` key. Returns `None` when
    `arr` is `None` (symbolic-only) or the dtype isn't one we model.

    Handles numpy / jax (`arr.dtype.name`) and PyTorch (same — Torch
    dtype objects also expose `.name`). FP8 variants are aliased to
    the OFP8 canonical name.
    """
    if arr is None:
        return None
    dt = getattr(arr, "dtype", None)
    if dt is None:
        return None
    name = getattr(dt, "name", None) or str(dt).rsplit(".", 1)[-1]
    name = _DTYPE_ALIASES.get(name, name)
    return name if name in MACHINE_EPS else None


def _as_aa(x) -> "AffineForm | None":
    """
    Normalize an operand to an `AffineForm | None`. Recognized inputs:
      - `AffineForm`: returned as-is.
      - Python `int` / `float`: wrapped as a zero-radius constant form.
      - Anything else (a typed value, a tracer, …): returns `None`.
    """
    if isinstance(x, AffineForm):
        return x
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return AffineForm.constant(float(x))
    return None


def _eps_for(out_dtype : "str | None") -> float:
    """
    Eps lookup with a fallback: an unknown dtype yields `0.0` (i.e.
    no rounding noise gets attached). The AA bound stays sound — the
    pre-rounding form already over-approximates — but is tighter than
    if we'd guessed a default. Callers that need a guaranteed
    rounding-noise contribution should supply a known dtype.
    """
    if out_dtype is None:
        return 0.0
    return MACHINE_EPS[out_dtype]


def compose_binary(
    op : str,
    lhs_aa : "AffineForm | None | int | float",
    rhs_aa : "AffineForm | None | int | float",
    out_dtype : "str | None",
) -> "AffineForm | None":
    """
    Compose AA for `lhs <op> rhs`, where `op` is one of `+`, `-`, `*`,
    `/`, `max`. Scalars are auto-wrapped into constant `AffineForm`s
    so `x + 3.0` works without the caller pre-wrapping the literal.

    `out_dtype` is the dtype of the *result* (which is what the FP-
    rounding noise attaches at). For numpy/jax/torch the backend
    handler computes `new_arr = lhs + rhs` and hands `_dtype_name(
    new_arr)` in — that's the dtype after the library's own type-
    promotion rules.

    Returns `None` if either side resolves to `None` (one operand is
    a tracer or has `.aa = None`), OR if the AA primitive raises a
    domain error (e.g. `div` with a divisor whose range crosses zero
    — common after a loose `exp` linearization). The graceful-
    degrade behavior is intentional: AA failure means the bound is
    unknown, not that the computation itself is invalid. Consumers
    check `result.aa is None` to detect "no bound available."
    """
    a = _as_aa(lhs_aa) if not isinstance(lhs_aa, AffineForm) else lhs_aa
    b = _as_aa(rhs_aa) if not isinstance(rhs_aa, AffineForm) else rhs_aa
    if a is None or b is None:
        return None
    eps = _eps_for(out_dtype)
    try:
        if op == "+":
            return round_fp(add(a, b), eps, label="add-round")
        if op == "-":
            return round_fp(sub(a, b), eps, label="sub-round")
        if op == "*":
            return round_fp(mul(a, b), eps, label="mul-round")
        if op == "/":
            return round_fp(div(a, b), eps, label="div-round")
        if op == "max":
            # `max` is a comparison + select; no FP rounding.
            return aa_maximum(a, b)
    except ValueError:
        # AA-domain errors (e.g. `1/x` over a range containing zero).
        # Real array op still succeeds; we just can't bound it.
        return None
    raise ValueError(f"compose_binary: unknown op {op!r}")


def compose_unary(
    op : str,
    child_aa : "AffineForm | None",
    out_dtype : "str | None",
) -> "AffineForm | None":
    """
    Compose AA for a unary math op (`exp`, `sin`, `cos`, `sqrt`).
    Linearization error lives in the operand's affine form; one more
    rounding noise is attached for the output op itself, scaled by
    `MACHINE_EPS[out_dtype]`.

    Returns `None` on AA-domain errors (e.g. `sqrt` over a range that
    extends below zero) — see `compose_binary` for the rationale.
    """
    if child_aa is None:
        return None
    eps = _eps_for(out_dtype)
    try:
        if op == "exp":
            return round_fp(aa_exp(child_aa), eps, label="exp-round")
        if op == "sqrt":
            return round_fp(aa_sqrt(child_aa), eps, label="sqrt-round")
        if op == "sin":
            return round_fp(
                affine_unary(child_aa, math.sin, math.cos, "sin-lin"),
                eps, label="sin-round",
            )
        if op == "cos":
            return round_fp(
                affine_unary(child_aa, math.cos, lambda x: -math.sin(x), "cos-lin"),
                eps, label="cos-round",
            )
    except ValueError:
        return None
    raise ValueError(f"compose_unary: unknown op {op!r}")


def sum_reduction(
    child_aa : "AffineForm | None",
    n : int,
    hardware : HardwareNumerics,
) -> "AffineForm | None":
    """
    AA for `Σ_{i in [0, n)} child` under the hardware's
    `reduction_order` + `accumulator_dtype`. Public-API form of the
    evaluator-internal `_sum_reduction`.
    """
    if child_aa is None:
        return None
    # Delegate to the evaluator's internal helper — same code path
    # walking ETs uses, so behavior stays consistent.
    return _sum_reduction(child_aa, n, hardware)


def _matmul_mul_eps(
    x_dtype : "str | None", y_dtype : "str | None",
    hardware : HardwareNumerics,
) -> float:
    """
    Eps for the matmul's multiplication step. Resolution order:

      1. `hardware.matmul_input_dtype` if explicitly set on the model —
         this captures hardware that forces a precision distinct from
         operand dtypes (TF32 multiplier on fp32 inputs; TPU MXU's
         internal bf16 even on bf16-or-wider inputs).
      2. The looser of the two operand dtypes — host promotion would
         widen to the more precise dtype, but the AA bound stays sound
         (over-approximates) using the wider eps.
      3. `hardware.default_eps` as the final fallback (operand dtypes
         unresolved, e.g. symbolic-only inputs).
    """
    if hardware.matmul_input_dtype is not None:
        return MACHINE_EPS[hardware.matmul_input_dtype]
    if x_dtype is not None and y_dtype is not None:
        return max(MACHINE_EPS[x_dtype], MACHINE_EPS[y_dtype])
    return hardware.default_eps


def compose_einsum(
    x_aa : "AffineForm | None",
    y_aa : "AffineForm | None",
    x_type, y_type, einstr : str,
    hardware : HardwareNumerics,
    *,
    x_dtype : "str | None" = None,
    y_dtype : "str | None" = None,
) -> "AffineForm | None":
    """
    Compose AA for `einsum(x, y, einstr)`. Modeled as one multiplication
    (at the matmul's mul-eps; see `_matmul_mul_eps` for resolution)
    followed by a reduction over the product of all contracted-dim
    sizes (at `hardware.acc_eps`, under `hardware.reduction_order`).

    Operand dtypes are passed in as `x_dtype` / `y_dtype` keyword args.
    When the active hardware model doesn't pin a `matmul_input_dtype`,
    the operand dtypes drive the multiplication's epsilon — so a host-
    side matmul of two bf16 tensors picks up bf16 rounding noise even
    under `WORST_CASE`. Backends call `dtype_name_of(x.arr)` /
    `dtype_name_of(y.arr)` and hand the result here.
    """
    if x_aa is None or y_aa is None:
        return None
    mul_eps = _matmul_mul_eps(x_dtype, y_dtype, hardware)
    # Parse the einstr to find which dims are contracted (= present in
    # the input sides but not in the output side). Sum-reduction count
    # is the product of those dims' sizes.
    lhs, rhs = einstr.split("->")
    lhs_x_dims = lhs.split(",")[0].strip().split()
    lhs_y_dims = lhs.split(",")[1].strip().split()
    out_dims = rhs.strip().split()
    input_dim_set = set(lhs_x_dims) | set(lhs_y_dims)
    contracted_dim_names = input_dim_set - set(out_dims)
    if not contracted_dim_names:
        # Pure pointwise einsum (outer product etc.): no reduction;
        # one mul + one rounding.
        return round_fp(
            mul(x_aa, y_aa), mul_eps,
            label="einsum-mul-round",
        )
    # Resolve contracted-dim names → sizes via the input Types.
    size_by_name = {}
    for d in (*x_type.st, *y_type.st):
        size_by_name[dim_name(d)] = as_int(dim_size(d))
    contraction_n = 1
    for name in contracted_dim_names:
        n = size_by_name.get(name)
        if n is None:
            # Can't resolve the contracted dim size → bail on AA.
            return None
        contraction_n *= n
    # One mul (at `mul_eps` — see `_matmul_mul_eps`), then sum over
    # contracted_n entries (at `acc_eps` via the reduction_order
    # policy).
    product = round_fp(
        mul(x_aa, y_aa), mul_eps,
        label="einsum-mul-round",
    )
    if contraction_n <= 1:
        return product
    return _sum_reduction(product, contraction_n, hardware)
