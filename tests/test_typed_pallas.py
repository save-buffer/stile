import sys

import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax

import stile
from stile import expr_simplifies, reset_stile, dim

@pytest.fixture
def reset():
    yield
    reset_stile()


def add_vectors_kernel(
    x : TypedRef,
    y : TypedRef,
    o : TypedOutputRef,
):
    o_ref.assign(x_ref + y_ref)


def test_add_vectors(reset):
    key = jax.random.PRNGKey(0)
    N = dim('N', 128)
    x = tjnp.random.normal(key, N)
    y = tjnp.random.normal(key, N)

    tpl.typed_pallas_call(
        add_vectors_kernel,
        out_type=stile.OutputSpec(
            'N + N',
            x.type.st,
            x.type.dt,
        )
    )(x, y)
