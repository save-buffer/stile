"""
Tests for stile's affine-arithmetic core.

The goal isn't full numerical-analysis coverage yet — it's pinning
the behaviors we'll lean on when wiring AA into the verifier:

  1. Linear ops preserve noise-symbol correlations (so `x - x` is
     literal zero, not `[-2|x|, 2|x|]`).
  2. Nonlinear ops introduce a single fresh noise symbol per
     application, bounding the linearization error.
  3. Range computation is conservative but tight on linear paths.
  4. Machine-epsilon table covers the dtypes a typed-Triton kernel
     can actually be launched with.
"""
import math

import pytest

import stile.numerical as nx
from stile.numerical import (
    AffineForm, fresh_noise,
    add, sub, neg, scale, mul, div, exp, sqrt, reciprocal,
    round_fp, MACHINE_EPS,
)


def test_constant_form_has_zero_radius(reset):
    c = AffineForm.constant(3.14)
    assert c.central == 3.14
    assert c.total_radius() == 0.0
    assert c.range() == (3.14, 3.14)


def test_x_minus_x_is_literal_zero(reset):
    """The headline AA property: shared noise symbols cancel exactly."""
    x = AffineForm.with_noise(central=5.0, symbol=fresh_noise("x"), coeff=2.0)
    diff = sub(x, x)
    assert diff.central == 0.0
    assert diff.noise == ()       # all coefficients dropped
    assert diff.range() == (0.0, 0.0)


def test_x_plus_y_minus_y_equals_x(reset):
    """A correlated round-trip: y cancels regardless of its own range."""
    ex, ey = fresh_noise("x"), fresh_noise("y")
    x = AffineForm.with_noise(2.0, ex, 1.0)
    y = AffineForm.with_noise(7.0, ey, 0.5)
    result = sub(add(x, y), y)
    assert result.central == x.central
    assert dict(result.noise) == dict(x.noise)


def test_range_is_central_plus_minus_total_radius(reset):
    e1, e2 = fresh_noise("a"), fresh_noise("b")
    x = AffineForm(central=10.0, noise=((e1, 2.0), (e2, -3.0)))
    assert x.total_radius() == 5.0
    assert x.range() == (5.0, 15.0)


def test_scale_is_linear(reset):
    ex = fresh_noise("x")
    x = AffineForm.with_noise(2.0, ex, 1.5)
    y = scale(x, -4.0)
    assert y.central == -8.0
    assert dict(y.noise) == {ex: -6.0}


def test_mul_preserves_linear_correlation_adds_cross_term(reset):
    """
    `(a₀ + a₁ε)(b₀ + b₂η) = a₀b₀ + a₀b₂·η + b₀a₁·ε + (cross)·new_ε`.
    The linear parts in `ε` and `η` are exact; the bilinear interaction
    `a₁b₂·εη` becomes a single fresh noise of magnitude `|a₁|·|b₂|`.
    """
    ex, ey = fresh_noise("x"), fresh_noise("y")
    a = AffineForm.with_noise(3.0, ex, 2.0)
    b = AffineForm.with_noise(5.0, ey, 1.0)
    prod = mul(a, b)
    assert prod.central == 15.0
    by_symbol = dict(prod.noise)
    # Linear coefficients in the original ε's:
    #   ε coeff = b_central · a_coeff = 5 · 2 = 10
    #   η coeff = a_central · b_coeff = 3 · 1 = 3
    assert by_symbol[ex] == 10.0
    assert by_symbol[ey] == 3.0
    # Plus exactly one fresh symbol for the cross-term, with
    # magnitude |a_radius| * |b_radius| = 2.0 * 1.0 = 2.0.
    cross = [(s, c) for s, c in prod.noise if s not in (ex, ey)]
    assert len(cross) == 1
    assert cross[0][1] == 2.0


def test_mul_with_constant_has_no_cross_term(reset):
    """Multiplying by a zero-radius form skips the fresh noise."""
    ex = fresh_noise("x")
    a = AffineForm.with_noise(3.0, ex, 2.0)
    k = AffineForm.constant(4.0)
    prod = mul(a, k)
    assert prod.central == 12.0
    assert dict(prod.noise) == {ex: 8.0}


def test_exp_linearization_bounds_the_curve(reset):
    """
    `exp(x)` for `x ∈ [0, 1]` central 0.5 — the linearization is a
    secant line; the gap is bounded by a fresh noise. Loosely
    sanity-check that the result range covers `[exp(0), exp(1)]`.
    """
    ex = fresh_noise("x")
    x = AffineForm.with_noise(0.5, ex, 0.5)   # x ∈ [0, 1]
    y = exp(x)
    lo, hi = y.range()
    assert lo <= math.exp(0.0)
    assert hi >= math.exp(1.0)


def test_reciprocal_rejects_range_through_zero(reset):
    ex = fresh_noise("x")
    x = AffineForm.with_noise(0.0, ex, 1.0)  # x ∈ [-1, 1]
    with pytest.raises(ValueError, match="range.*zero"):
        reciprocal(x)


def test_sqrt_rejects_negative_range(reset):
    ex = fresh_noise("x")
    x = AffineForm.with_noise(0.0, ex, 1.0)  # x ∈ [-1, 1]
    with pytest.raises(ValueError, match="below zero"):
        sqrt(x)


def test_round_fp_adds_bounded_relative_error(reset):
    """A single fp32 op should grow the radius by at most ε·|value|."""
    ex = fresh_noise("x")
    x = AffineForm.with_noise(3.0, ex, 0.1)
    eps = MACHINE_EPS["float32"]
    rounded = round_fp(x, eps)
    # Original radius: 0.1. New radius bounded by 0.1 + ε·max|x|.
    max_mag = max(abs(2.9), abs(3.1))  # range = [2.9, 3.1]
    expected_rad = 0.1 + eps * max_mag
    assert math.isclose(rounded.total_radius(), expected_rad, rel_tol=1e-9)


def test_machine_eps_covers_supported_dtypes(reset):
    """Every dtype a typed-Triton kernel can use should be listed."""
    expected = {
        "float64", "float32", "float16", "bfloat16", "tf32",
        "fp8_e3m4", "fp8_e4m3", "fp8_e5m2",
    }
    assert expected <= set(MACHINE_EPS)
    # fp8 has progressively wider unit roundoff as mantissa shrinks.
    assert MACHINE_EPS["fp8_e3m4"] < MACHINE_EPS["fp8_e4m3"] < MACHINE_EPS["fp8_e5m2"]
