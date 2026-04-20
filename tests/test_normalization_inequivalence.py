"""Tests that the normalizer correctly REJECTS non-equivalent expressions.

Positive equivalences are covered in the backend-specific tests. These tests
guard against the opposite failure mode: canonicalizing too aggressively and
accepting rewrites that aren't algebraically valid.
"""
import pytest

from stile import dim, reset_stile
from stile.specification import parse_spec_into_type
from stile.verification import verify_exprs_equivalent


@pytest.fixture
def reset():
    yield
    reset_stile()


def assert_not_equivalent(a : str, b : str):
    ta = parse_spec_into_type(a)
    tb = parse_spec_into_type(b)
    assert not verify_exprs_equivalent(ta.et, tb.et), (
        f"Expected {a!r} and {b!r} to normalize to different forms, "
        f"but the verifier considered them equivalent."
    )


def test_different_constants(reset):
    dim('N', 8)
    assert_not_equivalent("2 * N", "3 * N")


def test_different_tensors(reset):
    dim('M', 8)
    dim('N', 8)
    assert_not_equivalent("M", "N")


def test_softmax_temperature_is_nonlinear(reset):
    """softmax(x/c) is NOT softmax(x)/c — this is the bug we fixed in
    test_flash_attention's spec."""
    dim('N', 8)
    assert_not_equivalent(
        "softmax[N](N / 4)",
        "softmax[N](N) / 4",
    )


def test_subtraction_is_not_commutative(reset):
    dim('N', 8)
    assert_not_equivalent("N - 1", "1 - N")


def test_division_is_not_associative(reset):
    dim('N', 8)
    assert_not_equivalent("N / (2 / 3)", "(N / 2) / 3")


def test_exp_does_not_distribute_over_sum(reset):
    dim('N', 8)
    # exp(N+N) = exp(2N), not exp(N)+exp(N) (= 2*exp(N)).
    assert_not_equivalent("exp(N + N)", "exp(N) + exp(N)")


def test_sum_and_max_reductions_differ(reset):
    dim('N', 8)
    assert_not_equivalent("sum[N](N)", "max[N](N)")


def test_different_reduce_intervals(reset):
    dim('N', 8)
    assert_not_equivalent("sum[N](N[0:4])", "sum[N](N[0:8])")


def test_exp_is_not_the_identity(reset):
    dim('N', 8)
    # Guard against `exp(x) == x` being accepted for non-zero x.
    assert_not_equivalent("exp(N)", "N")


def test_sqrt_is_not_the_identity(reset):
    dim('N', 8)
    assert_not_equivalent("sqrt(N)", "N")


def test_exp_arg_matters(reset):
    """`exp(0) == 1` is a real rule, but `exp(N)` must not reduce to 1."""
    dim('N', 8)
    assert_not_equivalent("exp(N)", "1")


def test_sum_of_repeat_is_not_identity(reset):
    """sum[N](x) when x doesn't depend on N gives N.size * x, not x."""
    dim('N', 8)
    dim('M', 8)
    # sum over N of (M repeated over N) = size(N) * M = 8 * M, not M.
    assert_not_equivalent("sum[N](M -> N M)", "M")
