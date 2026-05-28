"""
Affine arithmetic — the numerical-bound primitive.

Each value is represented as a central magnitude plus a linear
combination of independent noise symbols `εᵢ ∈ [-1, 1]`:

    x = x₀ + Σᵢ xᵢ · εᵢ

Two key properties:

1. **Linear ops preserve the noise symbols exactly.** So `x - x` gives
   literal zero (every `εᵢ` coefficient cancels). Compare with interval
   arithmetic, which would give `[x_min - x_max, x_max - x_min]`.

2. **Nonlinear ops introduce a fresh noise symbol** whose magnitude
   bounds the linearization error. The information that an op's output
   came from a nonlinear step is preserved, but its specific
   correlation with prior `εᵢ`s is dropped (replaced by a worst-case
   independent term).

The range of an `AffineForm` is `[x₀ - Σ|xᵢ|, x₀ + Σ|xᵢ|]`.

This module is the substrate for stile's numerical analysis: every op
in a kernel attaches its own rounding-error `εᵢ` (bounded by the
dtype's machine epsilon), and two kernels that compute the same value
share leaf noises so that subtracting their `AffineForm`s cancels all
shared error and leaves only the per-op rounding gap.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field


# Module-level counter for fresh noise-symbol identities. Independent
# across processes — within a process, every call to `fresh_noise()`
# yields a unique tag.
_noise_counter = itertools.count()


@dataclass(frozen=True)
class NoiseSymbol:
    """
    An independent noise source. `id` is the unique tag; `label` is
    purely for debugging output. Equality + hashing use `id` only.
    """
    id : int
    label : str = ""

    def __eq__(self, other):
        return isinstance(other, NoiseSymbol) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


def fresh_noise(label : str = "") -> NoiseSymbol:
    """Allocate a new noise symbol with the given debug label."""
    return NoiseSymbol(id=next(_noise_counter), label=label)


@dataclass(frozen=True)
class AffineForm:
    """
    `central + Σ coeff·ε` over a set of `NoiseSymbol`s. Stored sparsely
    — symbols absent from `noise` have coefficient zero. Frozen so
    instances are hashable / shareable; ops construct fresh ones.
    """
    central : float
    noise : "tuple[tuple[NoiseSymbol, float], ...]" = field(default_factory=tuple)

    def __post_init__(self):
        # Canonicalize: sort by id, drop zero coefficients. Keeps
        # equality stable across construction orders.
        cleaned = tuple(
            (s, c) for s, c in self.noise if c != 0.0
        )
        cleaned = tuple(sorted(cleaned, key=lambda sc: sc[0].id))
        # Bypass frozen guard.
        object.__setattr__(self, "noise", cleaned)

    @classmethod
    def constant(cls, c : float) -> "AffineForm":
        return cls(central=float(c))

    @classmethod
    def with_noise(
        cls, central : float, symbol : NoiseSymbol, coeff : float,
    ) -> "AffineForm":
        return cls(central=float(central), noise=((symbol, float(coeff)),))

    def total_radius(self) -> float:
        """Sum of |coeff| over all noise symbols. Conservative upper
        bound on |x - central|."""
        return sum(abs(c) for _, c in self.noise)

    def range(self) -> "tuple[float, float]":
        r = self.total_radius()
        return (self.central - r, self.central + r)

    def _to_dict(self) -> dict:
        return dict(self.noise)


def _combine(
    a : AffineForm, b : AffineForm, op,
) -> dict:
    """Apply `op(a_coeff, b_coeff)` over the union of `a` and `b`'s
    noise symbols. Used for `+` (op=`x+y`) and `-` (op=`x-y`)."""
    out = {s: c for s, c in a.noise}
    for s, c in b.noise:
        out[s] = op(out.get(s, 0.0), c)
    return out


def add(a : AffineForm, b : AffineForm) -> AffineForm:
    """`(a + b) = (a₀+b₀) + Σ (aᵢ+bᵢ)·εᵢ`. Preserves correlations."""
    return AffineForm(
        central=a.central + b.central,
        noise=tuple(_combine(a, b, lambda x, y: x + y).items()),
    )


def sub(a : AffineForm, b : AffineForm) -> AffineForm:
    """`(a - b) = (a₀-b₀) + Σ (aᵢ-bᵢ)·εᵢ`. So `x - x = 0` exactly:
    each shared `εᵢ` cancels."""
    return AffineForm(
        central=a.central - b.central,
        noise=tuple(_combine(a, b, lambda x, y: x - y).items()),
    )


def neg(a : AffineForm) -> AffineForm:
    return AffineForm(
        central=-a.central,
        noise=tuple((s, -c) for s, c in a.noise),
    )


def scale(a : AffineForm, k : float) -> AffineForm:
    """Multiply by a scalar constant. Linear, no new noise."""
    return AffineForm(
        central=k * a.central,
        noise=tuple((s, k * c) for s, c in a.noise),
    )


def mul(a : AffineForm, b : AffineForm) -> AffineForm:
    """
    `(a₀ + Σaᵢεᵢ)(b₀ + Σbⱼεⱼ) = a₀b₀ + Σ(a₀bᵢ + b₀aᵢ)·εᵢ + (cross-terms)`

    The cross-terms `Σᵢⱼ aᵢbⱼ εᵢεⱼ` are bounded by `(Σ|aᵢ|)(Σ|bⱼ|)`;
    we collapse them into a single fresh noise symbol with that
    coefficient. The linear correlation between `a` and `b` against
    other forms is preserved; the bilinear interaction is bounded
    but no longer correlated with the inputs (this is the
    fundamental approximation in AA).
    """
    linear = {}
    for s, c in a.noise:
        linear[s] = linear.get(s, 0.0) + b.central * c
    for s, c in b.noise:
        linear[s] = linear.get(s, 0.0) + a.central * c
    cross = a.total_radius() * b.total_radius()
    noise = tuple(linear.items())
    if cross > 0.0:
        noise = noise + ((fresh_noise("mul-cross"), cross),)
    return AffineForm(
        central=a.central * b.central,
        noise=noise,
    )


def _min_max_over_range(
    f, df, a_lo : float, a_hi : float,
) -> "tuple[float, float, float]":
    """
    Build a linear envelope for a monotonic-derivative function
    `f` over `[a_lo, a_hi]`. Returns `(alpha, beta, delta)` such that
    `alpha·x + beta ± delta` upper/lower-bounds `f(x)` on the
    interval. The min-range form uses the line through the two
    endpoints; `delta` is the maximum gap between that secant and
    `f`. For `f` convex (e.g. `exp`), the max gap is at the point
    where `f'` equals the secant slope.
    """
    if a_lo == a_hi:
        v = f(a_lo)
        return (0.0, v, 0.0)
    f_lo, f_hi = f(a_lo), f(a_hi)
    alpha = (f_hi - f_lo) / (a_hi - a_lo)
    # Secant passes through (a_lo, f_lo); offset is f_lo - alpha*a_lo.
    beta = f_lo - alpha * a_lo
    # For convex `f`, the secant lies above f on the interval; max
    # gap is at the interior point where f'(x*) = alpha. Use df for
    # this and bound.
    try:
        # Solve df(x*) = alpha if possible by sampling; for `exp` and
        # `sqrt` we know x* analytically, but a coarse sample-based
        # bound is fine for the prototype.
        n_samples = 16
        gaps = []
        for i in range(n_samples + 1):
            t = a_lo + (a_hi - a_lo) * i / n_samples
            gaps.append(abs(f(t) - (alpha * t + beta)))
        delta = max(gaps)
    except Exception:
        delta = abs(f_hi - f_lo)
    return (alpha, beta, delta)


def affine_unary(
    a : AffineForm, f, df,
    label : str = "unary",
) -> AffineForm:
    """
    Generic min-range linearization for a unary function `f`. Computes
    `f(a) ≈ alpha·a + beta` over `a`'s range and bounds the gap with a
    fresh noise symbol.
    """
    a_lo, a_hi = a.range()
    alpha, beta, delta = _min_max_over_range(f, df, a_lo, a_hi)
    # Result = alpha·a + beta + delta·ε_new (the linearization error).
    result = scale(a, alpha)
    result = add(result, AffineForm.constant(beta))
    if delta > 0.0:
        result = add(
            result, AffineForm.with_noise(0.0, fresh_noise(label), delta),
        )
    return result


def exp(a : AffineForm) -> AffineForm:
    return affine_unary(a, math.exp, math.exp, label="exp-lin")


def sqrt(a : AffineForm) -> AffineForm:
    lo, hi = a.range()
    if lo < 0.0:
        raise ValueError(
            f"sqrt: input range {(lo, hi)} extends below zero — "
            f"AA can't bound on a non-monotone interval here."
        )
    return affine_unary(
        a, math.sqrt, lambda x: 0.5 / math.sqrt(x), label="sqrt-lin",
    )


def reciprocal(a : AffineForm) -> AffineForm:
    """`1 / a` via min-range linearization. Requires `a`'s range to
    not contain zero (otherwise `1/x` is unbounded)."""
    lo, hi = a.range()
    if lo <= 0.0 <= hi:
        raise ValueError(
            f"reciprocal: input range {(lo, hi)} contains zero."
        )
    return affine_unary(
        a, lambda x: 1.0 / x, lambda x: -1.0 / (x * x),
        label="recip-lin",
    )


def div(a : AffineForm, b : AffineForm) -> AffineForm:
    """`a / b` = `a · (1/b)`. Linearization error lives in the
    `reciprocal` step; `mul` adds the bilinear cross-term."""
    return mul(a, reciprocal(b))


def maximum(a : AffineForm, b : AffineForm) -> AffineForm:
    """
    Elementwise max via interval bounds:
        result ∈ [max(a_lo, b_lo), max(a_hi, b_hi)]
    Linearization is loose: we represent the result as the interval
    midpoint plus a fresh noise symbol covering the full range. This
    drops *all* prior correlations — `max(x, x)` ≠ `x` under this
    rule. A better treatment is on the todo list (e.g. piecewise
    affine over the regions where one side dominates).
    """
    a_lo, a_hi = a.range()
    b_lo, b_hi = b.range()
    lo = max(a_lo, b_lo)
    hi = max(a_hi, b_hi)
    mid = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)
    if rad == 0.0:
        return AffineForm.constant(mid)
    return AffineForm.with_noise(mid, fresh_noise("max-bound"), rad)


def round_fp(
    a : AffineForm, machine_eps : float, label : str = "rounding",
) -> AffineForm:
    """
    Model a single floating-point rounding step: the value is
    perturbed by at most `machine_eps * |value|`. Attaches a fresh
    noise symbol whose coefficient bounds that absolute error using
    the current range's maximum magnitude.
    """
    lo, hi = a.range()
    max_mag = max(abs(lo), abs(hi))
    eps_coeff = machine_eps * max_mag
    if eps_coeff == 0.0:
        return a
    return add(
        a, AffineForm.with_noise(0.0, fresh_noise(label), eps_coeff),
    )


# Machine epsilons for common dtypes — IEEE-754 unit-roundoff
# (`2^-(m+1)` where `m` is the explicit-mantissa-bit count, i.e. half
# the gap to the next representable in `[1, 2)`). For `round-to-
# nearest` rounding modes, the relative error of a single op is
# bounded by ε.
MACHINE_EPS = {
    "float64"  : 2 ** -53,   # 1 s, 11 e, 52 m
    "float32"  : 2 ** -24,   # 1 s,  8 e, 23 m
    "tf32"     : 2 ** -11,   # 1 s,  8 e, 10 m
    "float16"  : 2 ** -11,   # 1 s,  5 e, 10 m
    "bfloat16" : 2 ** -8,    # 1 s,  8 e,  7 m
    # FP8 family — non-standard in IEEE-754 but well-defined in
    # OFP8 / NVIDIA Hopper / OCP Microscaling docs. Unit roundoff
    # follows the same `2^-(m+1)` formula. E4M3 has no infinity
    # and S.1111.111 = NaN; the unit roundoff still holds in the
    # normal range.
    "fp8_e3m4" : 2 ** -5,    # 1 s, 3 e, 4 m
    "fp8_e4m3" : 2 ** -4,    # 1 s, 4 e, 3 m
    "fp8_e5m2" : 2 ** -3,    # 1 s, 5 e, 2 m
}
