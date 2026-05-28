"""
Numerical analysis for stile.

Affine arithmetic is the substrate: every value is a central magnitude
plus a linear combination of independent noise symbols
`εᵢ ∈ [-1, 1]`. Linear ops preserve the noise symbols exactly (so
`x - x` is literal zero); nonlinear ops introduce a fresh noise symbol
that bounds the linearization error.

Combined with the verifier's normalized form, this gives floating-point
analysis of expressions: each `tl.dot`, `+`, `exp`, etc. attaches its
own rounding-error `εᵢ` bounded by the dtype's machine epsilon, and
two kernels that compute the same mathematical value share leaf noises
so their AA difference cancels everything but the per-op rounding gap.
"""

from .affine import (
    AffineForm, NoiseSymbol, fresh_noise,
    add, sub, neg, scale, mul, div, exp, sqrt, reciprocal, maximum,
    round_fp, MACHINE_EPS,
)
from .hardware import (
    HardwareNumerics, ReductionOrder,
    WORST_CASE, NVIDIA_TENSOR_CORE_TF32, TRAINIUM_TENSOR_ENGINE, TPU_MXU,
    active_hardware, numerical_context,
)
from .evaluator import evaluate, exprs_close, rounding_error_bound
from .typed_ops import (
    compose_binary, compose_unary, compose_einsum, sum_reduction,
    dtype_name_of,
)
from .sensitivity import Sensitivity, sensitivity_analysis


def leaf_aa_from_array(arr) -> "AffineForm | None":
    """
    Build a leaf `AffineForm` from an actual array's `(min, max)`,
    centred at the midpoint with radius `(max - min) / 2` and a fresh
    noise symbol. Returns `None` when `arr` is `None` (symbolic-only
    typed values) or empty — both cases mean "no concrete data to
    bound from."

    Backends call this at construction of `TypedJaxArray` /
    `TypedTorchTensor` / etc. so every typed value enters the world
    with a sensible leaf AA, even when no per-op handler has updated
    it yet.
    """
    if arr is None:
        return None
    # Hostable on any array library that supports `.min()` / `.max()`
    # / `.item()`. NumPy, JAX, PyTorch all match.
    try:
        size = getattr(arr, "size", None)
        if size is not None and (callable(size) is False) and size == 0:
            return None
        lo = float(arr.min().item() if hasattr(arr.min(), "item") else arr.min())
        hi = float(arr.max().item() if hasattr(arr.max(), "item") else arr.max())
    except Exception:
        return None
    mid = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)
    if rad == 0.0:
        return AffineForm.constant(mid)
    return AffineForm.with_noise(mid, fresh_noise("leaf"), rad)
