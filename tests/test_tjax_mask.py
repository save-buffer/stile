"""
The `tjax.mask` intrinsic — a tagged constant tensor with user-chosen
in/out scalar values. Composes via the regular binary ops:
  - `score * mask(shape, p)` — mult-mask (the `.where(p)` sugar).
  - `score + mask(shape, p, 0.0, -jnp.inf)` — bias-mask for online
    softmax.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax
from stile import dim, reset_stile


@pytest.fixture
def reset():
    yield
    reset_stile()


def test_mult_mask_via_intrinsic(reset):
    qctx = dim('mqctx', 4)
    nctx = dim('mnctx', 4)
    m = tjax.mask((qctx, nctx), "mnctx <= mqctx")
    expected = jnp.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ], dtype=jnp.float32)
    assert jnp.array_equal(m.arr, expected)


def test_bias_mask_via_intrinsic(reset):
    qctx = dim('bqctx', 4)
    nctx = dim('bnctx', 4)
    m = tjax.mask((qctx, nctx), "bnctx <= bqctx", 0.0, -jnp.inf)
    # Inside the predicate: 0; outside: -inf.
    expected_inside = m.arr[jnp.tril_indices(4)]
    expected_outside = m.arr[jnp.triu_indices(4, k=1)]
    assert jnp.all(expected_inside == 0.0)
    assert jnp.all(jnp.isneginf(expected_outside))


def test_where_sugar_equivalent_to_explicit_mask(reset):
    """`.where(p)` and `self * tjax.mask(self.type.st, p)` produce the
    same TypedJaxArray (numerically and structurally)."""
    qctx = dim('wqctx', 4)
    nctx = dim('wnctx', 4)
    key = jax.random.PRNGKey(0)
    arr = tjax.random.normal(key, qctx, nctx)

    via_sugar = arr.where("wnctx <= wqctx")
    via_explicit = arr * tjax.mask(arr.type.st, "wnctx <= wqctx")

    assert jnp.array_equal(via_sugar.arr, via_explicit.arr)
    assert via_sugar.type.et == via_explicit.type.et


def test_mask_runtime_respects_sliced_shape(reset):
    """A mask built over a sliced shape evaluates the predicate at the
    *absolute* dim positions (q_real = q_slice + iqctx), not 0..len."""
    qctx = dim('sqctx', 8)
    nctx = dim('snctx', 8)
    iqctx, T = 4, 4
    sliced_shape = (qctx[iqctx:iqctx + T], nctx[:T])
    m = tjax.mask(sliced_shape, "snctx <= sqctx")
    # Tile covers q in [4,7], n in [0,3]. Every position is in-mask
    # (n <= q always) → all-ones.
    assert jnp.all(m.arr == 1.0)


def test_bias_form_equivalent_to_mult_form_under_exp(reset):
    """End-to-end: `exp(x) * mask(p)` and `exp(x + mask(p, 0, -inf))`
    normalize to the same expression via the bias→mult convergence
    landed in task #7."""
    from stile.verification import verify_exprs_equivalent

    qctx = dim('emq', 4)
    nctx = dim('emn', 4)
    key = jax.random.PRNGKey(0)
    x = tjax.random.normal(key, qctx, nctx)

    mult_form = tjax.exp(x) * tjax.mask(x.type.st, "emn <= emq")
    bias_form = tjax.exp(x + tjax.mask(x.type.st, "emn <= emq", 0.0, -jnp.inf))

    assert verify_exprs_equivalent(mult_form.type.et, bias_form.type.et)
