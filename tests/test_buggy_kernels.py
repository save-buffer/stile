"""
Buggy-kernel suite: kernels we deliberately get *wrong* and assert the
verifier rejects. Each test encodes one realistic mistake a kernel
author might make and pins down what stops it from sneaking past.

Organized by *category of bug the verifier catches*:

  - Arithmetic   — wrong constants, wrong ops, double-applied factors.
  - Reductions   — wrong reduce op, wrong dim, partial reduction.
  - Contractions — wrong contracted dim, wrong output shape.
  - Slicing      — wrong slice bounds.
  - Masking      — missing/flipped causal mask.
  - Coverage     — skipped tile, overlapping tiles.

Positive tests live elsewhere; this file is *only* about over-acceptance
— the worst failure mode for a verifier.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax
from stile import dim, reset_stile
from stile.specification import parse_spec_into_type
from stile.verification import verify_exprs_equivalent


@pytest.fixture
def reset():
    yield
    reset_stile()


def _expect_rejection(kernel_call, *args, **kwargs):
    """Run `kernel_call(*args, **kwargs)` and assert it raises (the
    verifier rejected it). The two assertion shapes that bubble up are
    `assert_equivalent`'s `AssertionError` and `assign`'s `ValueError`."""
    with pytest.raises((AssertionError, ValueError)):
        kernel_call(*args, **kwargs)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

def test_extra_constant_factor_rejected(reset):
    """Spec says `M N`, kernel produces `2 * M N`. Verifier rejects."""
    M, N = dim("ARM", 8), dim("ARN", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), M, N)
    buggy = a * 2
    _expect_rejection(buggy.assert_equivalent, "ARM ARN")


def test_missing_constant_factor_rejected(reset):
    """Spec says `2 * M N`, kernel produces just `M N`."""
    M, N = dim("MCM", 8), dim("MCN", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), M, N)
    _expect_rejection(a.assert_equivalent, "2 * MCM MCN")


def test_double_applied_factor_in_matmul_rejected(reset):
    """Spec says `2 * (M N, N K -> M K)` but kernel multiplies BOTH
    operands by 2 — yields a 4× result. The classic bug from the README
    that motivated stile."""
    M, N, K = dim("DAM", 8), dim("DAN", 8), dim("DAK", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), M, N)
    b = tjax.random.normal(jax.random.PRNGKey(1), N, K)
    buggy = tjax.einsum(a * 2, b * 2, "DAM DAN, DAN DAK -> DAM DAK")
    _expect_rejection(buggy.assert_equivalent, "2 * (DAM DAN, DAN DAK -> DAM DAK)")


def test_wrong_binary_op_rejected(reset):
    """Spec says addition, kernel does multiplication."""
    M, N = dim("WBM", 8), dim("WBN", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), M, N)
    b = tjax.random.normal(jax.random.PRNGKey(1), M, N)
    buggy = a * b
    _expect_rejection(buggy.assert_equivalent, "WBM WBN + WBM WBN")


def test_subtraction_not_commutative(reset):
    """At the spec level `N - 1` and `1 - N` must not normalize equal
    (sanity-pin from the spec inequivalence suite — kernel-level can't
    distinguish two tensors of the same dim signature, so we keep this
    pin spec-only)."""
    dim("SUN", 8)
    a = parse_spec_into_type("SUN - 1")
    b = parse_spec_into_type("1 - SUN")
    assert not verify_exprs_equivalent(a.et, b.et)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

def test_max_in_place_of_sum_rejected(reset):
    """Spec wants a sum reduction but kernel computes max."""
    N = dim("MSN", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), N)
    buggy = a.max(N)
    _expect_rejection(buggy.assert_equivalent, "sum[MSN](MSN) -> ")


def test_sum_in_place_of_max_rejected(reset):
    """Spec wants max but kernel computes sum."""
    N = dim("SMN", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), N)
    buggy = a.sum(N)
    _expect_rejection(buggy.assert_equivalent, "max[MSN](MSN) -> ")


def test_reduction_over_wrong_dim_rejected(reset):
    """Spec sums over N, kernel sums over M — different reduction dim."""
    M, N = dim("RWM", 8), dim("RWN", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), M, N)
    buggy = a.sum(M)  # reduces M instead of N
    _expect_rejection(buggy.assert_equivalent, "sum[RWN](RWM RWN) -> RWM")


# ---------------------------------------------------------------------------
# Contractions
# ---------------------------------------------------------------------------

def test_einsum_wrong_contracted_dim_rejected(reset):
    """Matmul spec contracts N. Buggy einsum contracts K instead — gives
    a totally different result of shape (M, N)."""
    M, N, K = dim("ECM", 8), dim("ECN", 8), dim("ECK", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), M, N)
    b = tjax.random.normal(jax.random.PRNGKey(1), N, K)
    # Correct matmul reduces N. Buggy: instead of contracting N we keep
    # it and contract over a different dim signature.
    buggy = tjax.einsum(a, b, "ECM ECN, ECN ECK -> ECM ECN")
    _expect_rejection(
        buggy.assert_equivalent,
        "(ECM ECN, ECN ECK -> ECM ECK)",
    )


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

def test_wrong_slice_bound_rejected(reset):
    """Spec sums over N[0:8], kernel sums over only N[0:4]."""
    N = dim("WSN", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), N)
    half = a.slice(N, 0, 4)
    buggy = half.sum(N)
    _expect_rejection(buggy.assert_equivalent, "sum[WSN](WSN) -> ")


def test_offset_assign_to_wrong_position_rejected(reset):
    """A kernel that writes a tile to the wrong output position is
    caught by `TypedResult.done()` — the slice bounds on the output
    are tracked, so a kernel that fills `[4:8]` instead of `[0:8]`
    leaves a gap. Slice info on *input* reads doesn't show up at the
    expression level (two slices of the same dim parse to the same
    leaf tensor), so this kind of bug is caught at the output side via
    coverage rather than via algebraic equivalence."""
    N = dim("OSN", 8)
    a = tjax.random.normal(jax.random.PRNGKey(0), N)
    result = tjax.TypedResult("OSN")
    # Mistake: only fill the upper half; lower half [0:4) untouched.
    result.assign(a.slice(N, 4, 8))
    with pytest.raises(ValueError, match="gap or overlap|only covered"):
        result.done()


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def test_causal_kernel_without_mask_rejected(reset):
    """Causal-attention spec, kernel forgets the mask. Unmasked attention
    must not satisfy the causal spec."""
    Q, K, D = dim("CKQ", 8), dim("CKK", 8), dim("CKD", 4)
    q = tjax.random.normal(jax.random.PRNGKey(0), Q, D)
    k = tjax.random.normal(jax.random.PRNGKey(1), K, D)
    # No mask applied:
    qk = tjax.einsum(q, k, "CKQ CKD, CKK CKD -> CKQ CKK")
    buggy = qk.sum(K)
    _expect_rejection(
        buggy.assert_equivalent,
        "sum[CKK where CKK <= CKQ](CKQ CKD, CKK CKD -> CKQ CKK) -> CKQ",
    )


def test_causal_predicate_flipped_rejected(reset):
    """Spec is `K <= Q` (causal); kernel applies `K >= Q` (anti-causal).
    Distinct triangles, distinct results."""
    Q, K, D = dim("FPQ", 8), dim("FPK", 8), dim("FPD", 4)
    q = tjax.random.normal(jax.random.PRNGKey(0), Q, D)
    k = tjax.random.normal(jax.random.PRNGKey(1), K, D)
    qk = tjax.einsum(q, k, "FPQ FPD, FPK FPD -> FPQ FPK")
    # Anti-causal mask:
    buggy = qk.where("FPK >= FPQ").sum(K)
    _expect_rejection(
        buggy.assert_equivalent,
        "sum[FPK where FPK <= FPQ](FPQ FPD, FPK FPD -> FPQ FPK) -> FPQ",
    )


def test_mask_in_place_of_bias_for_max_rejected(reset):
    """For a max-reduction, mult-mask gives the wrong answer (max sees
    `0` from masked positions). Verifier should reject when the kernel
    uses mult-mask but the spec uses `max[d where P]` (which lowers to
    bias-form). The two are mathematically distinct."""
    Q, K = dim("MBQ", 8), dim("MBK", 8)
    qk = tjax.random.normal(jax.random.PRNGKey(0), Q, K)
    # Buggy: multiplicative mask before max — masked positions become 0.
    buggy = qk.where("MBK <= MBQ").max(K)
    _expect_rejection(
        buggy.assert_equivalent,
        "max[MBK where MBK <= MBQ](MBQ MBK) -> MBQ",
    )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_skipped_tile_coverage_rejected(reset):
    """`TypedResult.done()` rejects when the kernel only assigned part
    of the output."""
    M = dim("STM", 16)
    a = tjax.random.normal(jax.random.PRNGKey(0), M)
    result = tjax.TypedResult("STM")
    # Only assign the first half — second half is uncovered.
    result.assign(a.slice(M, 0, 8))
    with pytest.raises(ValueError, match="only covered up to"):
        result.done()


def test_overlapping_tile_coverage_rejected(reset):
    """`TypedResult.done()` rejects when two assigns overlap on a dim
    (one tile written twice). Same-bounds duplicates are deduped, but
    a partial overlap is a real bug."""
    M = dim("OTM", 16)
    a = tjax.random.normal(jax.random.PRNGKey(0), M)
    result = tjax.TypedResult("OTM")
    result.assign(a.slice(M, 0, 8))
    result.assign(a.slice(M, 4, 16))  # overlaps [4, 8) with the first
    with pytest.raises(ValueError, match="gap or overlap"):
        result.done()


def test_gap_between_tiles_rejected(reset):
    """`TypedResult.done()` rejects a gap between non-adjacent assigns."""
    M = dim("GTM", 16)
    a = tjax.random.normal(jax.random.PRNGKey(0), M)
    result = tjax.TypedResult("GTM")
    result.assign(a.slice(M, 0, 4))
    result.assign(a.slice(M, 8, 16))  # leaves [4, 8) uncovered
    with pytest.raises(ValueError, match="gap or overlap"):
        result.done()


# ---------------------------------------------------------------------------
# Flash attention with subtle bugs
# ---------------------------------------------------------------------------
#
# Each test below is the working flash-attention kernel with ONE realistic
# mistake. Same correct spec on every test. The verifier should reject
# all of them — these pin down what stile catches that a numerical-only
# test might miss (or might catch with a flaky `allclose` on rare seeds).

_FA_SPEC = (
    "(softmax[FAnctx](("
    "FAqctx FAdhead, FAnctx FAdhead -> FAqctx FAnctx"
    ") / sqrt(16)), "
    "FAnctx FAdhead -> FAqctx FAdhead)"
)


def _flash_kernel(Q, K, V, qctx, nctx, dhead, *,
                  scale_factor=None,
                  exp_on_rescaling=True,
                  forgot_running_l_rescale=False,
                  nctx_tile_size=None):
    """A parametric flash-attention kernel. The keyword args toggle each
    subtle bug; with all defaults it's the correct kernel.

      - `scale_factor`: divisor for the QK^T tile. Correct = `sqrt(dhead.size)`.
      - `exp_on_rescaling`: whether to wrap rescaling in `exp()`. Correct = True.
      - `forgot_running_l_rescale`: skip applying the `exp(running_max -
        new_max)` correction to the running denominator on each merge.
        Only misbehaves once the loop has more than one iteration.
    """
    if scale_factor is None:
        scale_factor = jnp.sqrt(dhead.size)
    if nctx_tile_size is None:
        nctx_tile_size = nctx.size
    qctx_tile_size = qctx.size
    iqctx = 0
    running_max = -jnp.inf
    running_l = 0
    o = 0
    for ictx in range(0, nctx.size, nctx_tile_size):
        q_tile = Q.slice(qctx, iqctx, iqctx + qctx_tile_size)
        k_tile = K.slice(nctx, ictx, ictx + nctx_tile_size)
        qk_tile = tjax.einsum(
            q_tile, k_tile, "FAqctx FAdhead, FAnctx FAdhead -> FAnctx FAqctx",
        ) / scale_factor
        tile_max = qk_tile.max(nctx)
        logits = tjax.exp(qk_tile - tile_max.repeat(qk_tile.type.st[0]))
        tile_l = logits.sum(nctx)
        new_max = tjax.maximum(tile_max, running_max)

        if exp_on_rescaling:
            old_factor = tjax.exp(running_max - new_max)
            new_factor = tjax.exp(tile_max - new_max)
        else:
            # Bug: forgot to wrap in `exp`.
            old_factor = running_max - new_max
            new_factor = tile_max - new_max

        if forgot_running_l_rescale:
            # Bug: running_l carries over un-corrected.
            new_l = running_l + new_factor * tile_l
        else:
            new_l = old_factor * running_l + new_factor * tile_l

        v_tile = V.slice(nctx, ictx, ictx + nctx_tile_size)
        v_proj = tjax.einsum(
            logits, v_tile, "FAnctx FAqctx, FAnctx FAdhead -> FAqctx FAdhead",
        )

        rescaled_old_o = (
            running_l * old_factor
        ).repeat(dhead).rearrange(qctx, dhead) * o
        rescaled_v_proj = new_factor.repeat(dhead).rearrange(qctx, dhead) * v_proj
        o = (rescaled_old_o + rescaled_v_proj) / new_l.repeat(
            dhead,
        ).rearrange(qctx, dhead)
        running_l = new_l
        running_max = new_max
    return o


def _fa_setup():
    dhead = dim("FAdhead", 16)
    qctx = dim("FAqctx", 16)
    nctx = dim("FAnctx", 16)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = tjax.random.normal(k1, qctx, dhead)
    K = tjax.random.normal(k2, nctx, dhead)
    V = tjax.random.normal(k3, nctx, dhead)
    return Q, K, V, qctx, nctx, dhead


def test_flash_attention_correct_baseline(reset):
    """Sanity baseline: with all flags at their default (correct) values,
    the kernel verifies. If this fails, one of the buggy tests below
    might be passing for the wrong reason."""
    Q, K, V, qctx, nctx, dhead = _fa_setup()
    o = _flash_kernel(Q, K, V, qctx, nctx, dhead)
    o.assert_equivalent(_FA_SPEC)


def test_flash_attention_wrong_sqrt_divisor_rejected(reset):
    """User's suggested bug: divide by `sqrt(dhead - 1)` instead of
    `sqrt(dhead)`. Spec uses `sqrt(16)`; buggy kernel uses `sqrt(15)`."""
    Q, K, V, qctx, nctx, dhead = _fa_setup()
    o = _flash_kernel(
        Q, K, V, qctx, nctx, dhead,
        scale_factor=jnp.sqrt(dhead.size - 1),
    )
    _expect_rejection(o.assert_equivalent, _FA_SPEC)


def test_flash_attention_missing_exp_on_rescaling_rejected(reset):
    """Common typo: rescaling the running aggregates by a raw difference
    `(running_max - new_max)` instead of `exp(running_max - new_max)`.
    The kernel produces a totally different value but the structure of
    the ops looks tantalizingly close."""
    Q, K, V, qctx, nctx, dhead = _fa_setup()
    o = _flash_kernel(Q, K, V, qctx, nctx, dhead, exp_on_rescaling=False)
    _expect_rejection(o.assert_equivalent, _FA_SPEC)


def test_flash_attention_forgot_running_l_rescale_rejected(reset):
    """Bug: the running denominator is carried over without applying
    the `exp(running_max - new_max)` correction on each tile merge.
    Iteration 1 is identical to the correct kernel (running_l = 0
    initially, so the missing factor doesn't matter); iteration 2+
    diverges. Run with two `nctx` tiles so the bug actually exercises."""
    Q, K, V, qctx, nctx, dhead = _fa_setup()
    o = _flash_kernel(
        Q, K, V, qctx, nctx, dhead,
        forgot_running_l_rescale=True,
        nctx_tile_size=nctx.size // 2,
    )
    _expect_rejection(o.assert_equivalent, _FA_SPEC)
