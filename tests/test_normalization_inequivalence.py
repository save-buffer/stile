"""
Tests that the normalizer correctly REJECTS non-equivalent expressions.

Positive equivalences are covered in the backend-specific tests. These tests
guard against the opposite failure mode: canonicalizing too aggressively and
accepting rewrites that aren't algebraically valid.
"""
import pytest

from stile import dim, reset_stile, diff_exprs
from stile.specification import parse_spec_into_type
from stile.verification import verify_exprs_equivalent


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
    """
    softmax(x/c) is NOT softmax(x)/c — this is the bug we fixed in
    test_flash_attention's spec.
    """
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


# ---------------------------------------------------------------------------
# Tensor-identity-by-name (#16). The soundness contract: two distinct
# named tensors with the same shape do NOT collapse to one. Pinned here
# so any future regression in `Tensor.name` propagation surfaces loudly.
# ---------------------------------------------------------------------------

def test_distinct_labels_do_not_collapse_under_addition(reset):
    """
    `x + y` is NOT `2 * x` when `x` and `y` are distinct labeled
    tensors. The flaw before #16 silently accepted this; the fix makes
    name part of the leaf identity.
    """
    dim('N', 8)
    assert_not_equivalent("x:N + y:N", "2 * x:N")


def test_distinct_labels_are_distinct_leaves(reset):
    """
    Two labeled tensors with the same dim signature but different
    labels are different tensors.
    """
    dim('N', 8)
    assert_not_equivalent("x:N", "y:N")


def test_anonymous_occurrences_do_not_collapse(reset):
    """
    Two unlabeled tensor references in the same spec are distinct
    anonymous tensors — an unlabeled `N + N` is NOT `2 * N`. (For
    sharing, the user must use an explicit label.)
    """
    dim('N', 8)
    assert_not_equivalent("N + N", "2 * N")


def test_same_label_is_the_same_leaf(reset):
    """
    The contract goes both ways: when the user *does* use the same
    label, those references *do* collapse. `x + x` ≡ `2 * x`.
    """
    dim('N', 8)
    a = parse_spec_into_type("x:N + x:N")
    b = parse_spec_into_type("2 * x:N")
    assert verify_exprs_equivalent(a.et, b.et)


# --- diff_exprs: where two inequivalent ETs structurally diverge -------
# The companion to the rejection tests above: when `verify_exprs_equivalent`
# says False, `diff_exprs` reports the first structural divergence.

def test_diff_exprs_reports_wrong_dim(reset):
    """A transposed / wrong dim shows up as a dims-path divergence."""
    dim("M", 8); dim("N", 4); dim("K", 6)
    a = parse_spec_into_type("A:M N")
    b = parse_spec_into_type("A:M K")
    msg = diff_exprs(a.et, b.et)
    assert msg is not None
    assert ".dims[1]" in msg and "N" in msg and "K" in msg


def test_diff_exprs_reports_wrong_op(reset):
    """A `+` vs `*` divergence is reported at `.op`."""
    dim("N", 4)
    a = parse_spec_into_type("X:N + Y:N")
    b = parse_spec_into_type("X:N * Y:N")
    msg = diff_exprs(a.et, b.et)
    assert msg == "at .op: '+' vs '*'"


def test_diff_exprs_reports_wrong_constant(reset):
    """A wrong scalar coefficient surfaces at the constant operand."""
    dim("N", 4)
    a = parse_spec_into_type("2 * X:N")
    b = parse_spec_into_type("3 * X:N")
    msg = diff_exprs(a.et, b.et)
    assert msg is not None and "2.0" in msg and "3.0" in msg


def test_diff_exprs_none_for_commutative_reorder(reset):
    """`a * b` vs `b * a` are equivalent — no spurious diff."""
    dim("N", 4)
    a = parse_spec_into_type("X:N * Y:N")
    b = parse_spec_into_type("Y:N * X:N")
    assert diff_exprs(a.et, b.et) is None


def test_diff_exprs_none_for_equivalent_exprs(reset):
    """
    When the verifier says equivalent, diff returns None — nothing to
    chase.
    """
    dim("N", 4)
    a = parse_spec_into_type("X:N + Y:N")
    b = parse_spec_into_type("Y:N + X:N")
    assert diff_exprs(a.et, b.et) is None


def test_diff_exprs_commutative_aligns_swapped_operands(reset):
    """
    With check_equivalent=False, a commutative node still aligns
    operands order-insensitively, so the diff points at the genuinely
    different leaf rather than a phantom lhs/rhs swap.
    """
    dim("N", 4)
    # `X * Z` vs `Z * Y`: aligned by the matching `Z`, the real diff is
    # X vs Y, not a swap.
    a = parse_spec_into_type("X:N * Z:N")
    b = parse_spec_into_type("Z:N * Y:N")
    msg = diff_exprs(a.et, b.et, check_equivalent=False)
    assert msg is not None and "X" in msg and "Y" in msg


def test_diff_exprs_reports_node_type_mismatch(reset):
    """Different node kinds at the same position report a type mismatch."""
    dim("N", 4)
    a = parse_spec_into_type("X:N + Y:N")   # BinaryOp at root
    b = parse_spec_into_type("X:N")         # Tensor at root
    msg = diff_exprs(a.et, b.et, check_equivalent=False)
    assert msg is not None and "BinaryOp" in msg and "Tensor" in msg
