"""
Scatter-with-base (KV-cache writeback) tests.

`Scatter` carries an explicit `base`: the value at output positions the
scatter didn't address. `base=0` (the default) is the original
"scatter into zeros" behavior; a non-zero base expresses a writeback —
"keep the base everywhere the scatter didn't touch." This is what
paged-attention / streaming-decode KV-cache updates need, and what the
original `Scatter` (zero-filled) couldn't express.

Two surface syntaxes lower to the same `Scatter` ET:
  - function form:  `scatter[dim](values, idx, base)`
  - TLA+-style:     `base except [dim @ idx] = values`

The verifier treats scatter opaquely (matches by idx-tensor name), so
the writeback verifies structurally just like plain scatter — now with
the base threaded into the identity.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax
from stile import dim
from stile.specification import parse_spec_into_type
from stile.verification import verify_exprs_equivalent


def test_writeback_base_preserved_structural_equality(reset):
    """Two writebacks with identical values, dim, idx, and base are
    structurally equal."""
    M = dim("M", 8)
    N = dim("N", 32)
    D = dim("D", 4)
    new_vals = tjax.random.normal(jax.random.PRNGKey(0), M, D, name="V")
    cache = tjax.random.normal(jax.random.PRNGKey(1), N, D, name="C")
    idx = tjax.runtime_index("idx", M, values_in=N)
    a = new_vals.scatter(N, idx, base=cache)
    b = new_vals.scatter(N, idx, base=cache)
    assert verify_exprs_equivalent(a.type.et, b.type.et)


def test_writeback_differs_from_zero_scatter(reset):
    """A writeback (base = cache) is NOT equivalent to a plain scatter
    (base = 0). The base is part of the scatter's identity — collapsing
    them would be unsound (a plain scatter zeroes the untouched cache).
    """
    M = dim("M", 8)
    N = dim("N", 32)
    D = dim("D", 4)
    new_vals = tjax.random.normal(jax.random.PRNGKey(0), M, D, name="V")
    cache = tjax.random.normal(jax.random.PRNGKey(1), N, D, name="C")
    idx = tjax.runtime_index("idx", M, values_in=N)
    writeback = new_vals.scatter(N, idx, base=cache)
    zero_scatter = new_vals.scatter(N, idx)
    assert not verify_exprs_equivalent(writeback.type.et, zero_scatter.type.et)


def test_writeback_function_syntax_matches_programmatic(reset):
    """`scatter[N](V, idx, C)` in the spec language produces the same
    `Scatter` ET as the programmatic `.scatter(N, idx, base=C)`."""
    M = dim("M", 8)
    N = dim("N", 32)
    D = dim("D", 4)
    new_vals = tjax.random.normal(jax.random.PRNGKey(0), M, D, name="V")
    cache = tjax.random.normal(jax.random.PRNGKey(1), N, D, name="C")
    idx = tjax.runtime_index("idx", M, values_in=N)
    programmatic = new_vals.scatter(N, idx, base=cache)
    spec = parse_spec_into_type(
        "scatter[N](V:M D, idx:M, C:N D) -> N D"
    )
    assert verify_exprs_equivalent(programmatic.type.et, spec.et)


def test_writeback_except_syntax_matches_programmatic(reset):
    """The TLA+-style `C except [N @ idx] = V` lowers to the same
    `Scatter` ET as the programmatic writeback."""
    M = dim("M", 8)
    N = dim("N", 32)
    D = dim("D", 4)
    new_vals = tjax.random.normal(jax.random.PRNGKey(0), M, D, name="V")
    cache = tjax.random.normal(jax.random.PRNGKey(1), N, D, name="C")
    idx = tjax.runtime_index("idx", M, values_in=N)
    programmatic = new_vals.scatter(N, idx, base=cache)
    spec = parse_spec_into_type(
        "C:N D except [N @ idx:M] = V:M D -> N D"
    )
    assert verify_exprs_equivalent(programmatic.type.et, spec.et)


def test_except_and_function_syntax_agree(reset):
    """The two surface syntaxes are interchangeable — same ET."""
    M = dim("M", 8)
    N = dim("N", 32)
    D = dim("D", 4)
    fn_form = parse_spec_into_type("scatter[N](V:M D, idx:M, C:N D) -> N D")
    except_form = parse_spec_into_type("C:N D except [N @ idx:M] = V:M D -> N D")
    assert verify_exprs_equivalent(fn_form.et, except_form.et)


def test_writeback_distinct_base_not_equal(reset):
    """Two writebacks differing only in their base are distinct."""
    M = dim("M", 8)
    N = dim("N", 32)
    D = dim("D", 4)
    new_vals = tjax.random.normal(jax.random.PRNGKey(0), M, D, name="V")
    cache1 = tjax.random.normal(jax.random.PRNGKey(1), N, D, name="C1")
    cache2 = tjax.random.normal(jax.random.PRNGKey(2), N, D, name="C2")
    idx = tjax.runtime_index("idx", M, values_in=N)
    wb1 = new_vals.scatter(N, idx, base=cache1)
    wb2 = new_vals.scatter(N, idx, base=cache2)
    assert not verify_exprs_equivalent(wb1.type.et, wb2.type.et)


def test_writeback_permutation_round_trip_ignores_base(reset):
    """`gather(scatter(V, perm, base=anything), perm) = V` when `perm`
    is a permutation: a permutation overwrites every output position,
    so the base never shows through and the round-trip recovers V
    regardless of base."""
    N = dim("N", 16)
    D = dim("D", 4)
    V = tjax.random.normal(jax.random.PRNGKey(0), N, D, name="V")
    cache = tjax.random.normal(jax.random.PRNGKey(1), N, D, name="C")
    perm = tjax.runtime_index("perm", N, values_in=N, permutation=True)
    round_trip = V.scatter(N, perm, base=cache).gather(N, perm)
    assert verify_exprs_equivalent(round_trip.type.et, V.type.et)
