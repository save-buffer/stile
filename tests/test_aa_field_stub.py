"""
Stub-level tests for the always-on `aa` field on typed values.

At this stage (step 4 of the eager-AA rollout), each typed value
(`TypedJaxArray`, `TypedTorchTensor`, `TypedNumpyArray`) carries an
`aa : AffineForm | None` that's auto-populated at construction from
the actual array's `(min, max)`. Per-op handlers (step 5) will compose
new AAs through ops; today, the field on derived values is whatever
the op handler happened to set (usually nothing — falls back to a
fresh leaf form derived from the result tensor).

We verify here that the field is plumbed end-to-end and that
`leaf_aa_from_array` does the right thing on inputs with known ranges.
"""
import pytest
import numpy as np

from conftest import REQUIRES_TORCH, REQUIRES_JAX

import stile
from stile import dim
from stile.numerical import AffineForm, leaf_aa_from_array
from stile.numpy._core import TypedNumpyArray
from stile.type import Tensor, Type


def test_leaf_aa_from_array_centers_and_widens(reset):
    """
    A numpy array with known (min, max) becomes a leaf AffineForm
    with that midpoint + half-range radius.
    """
    arr = np.array([-2.0, 0.0, 4.0])
    form = leaf_aa_from_array(arr)
    assert isinstance(form, AffineForm)
    # min=-2, max=4 → mid=1, rad=3.
    assert form.central == 1.0
    assert form.range() == (-2.0, 4.0)


def test_leaf_aa_from_array_constant_when_no_spread(reset):
    """
    All-equal arrays collapse to a constant (zero-radius) form.
    """
    arr = np.full((5,), 7.0)
    form = leaf_aa_from_array(arr)
    assert form.central == 7.0
    assert form.total_radius() == 0.0


def test_leaf_aa_from_array_returns_none_on_none():
    """
    `None` array → `None` AA (symbolic-only typed values).
    """
    assert leaf_aa_from_array(None) is None


def test_typed_numpy_array_aa_auto_populated(reset):
    """
    A `TypedNumpyArray` constructor with an array should set `.aa`
    to a leaf form derived from the array's range.
    """
    N = dim("AaNpN", 4)
    arr = np.array([-1.0, 0.5, 2.0, 3.5])
    tnp = TypedNumpyArray(
        arr, Type(st=(N,), et=Tensor(dims=(N,), name="X")),
    )
    assert tnp.aa is not None
    assert tnp.aa.central == 1.25  # midpoint of [-1, 3.5]
    assert tnp.aa.range() == (-1.0, 3.5)


def test_typed_numpy_array_aa_opt_out_via_explicit_none(reset):
    """
    Passing `aa=None` explicitly opts out of leaf computation.
    """
    N = dim("AaNpN2", 4)
    arr = np.array([0.0, 1.0, 2.0, 3.0])
    tnp = TypedNumpyArray(
        arr, Type(st=(N,), et=Tensor(dims=(N,), name="X")),
        aa=None,
    )
    assert tnp.aa is None


@REQUIRES_JAX
def test_typed_jax_array_aa_auto_populated(reset):
    """
    A `TypedJaxArray` constructor with an array should set `.aa`.
    """
    import jax.numpy as jnp
    from stile.jax._core import TypedJaxArray
    N = dim("AaJxN", 4)
    arr = jnp.array([-1.0, 0.5, 2.0, 3.5])
    tj = TypedJaxArray(
        arr, Type(st=(N,), et=Tensor(dims=(N,), name="X")),
    )
    assert tj.aa is not None
    assert tj.aa.range() == (-1.0, 3.5)


@REQUIRES_JAX
def test_typed_jax_array_aa_none_when_arr_is_none(reset):
    """
    Symbolic-only TypedJaxArray (arr=None) has aa=None.
    """
    from stile.jax._core import TypedJaxArray
    N = dim("AaJxN3", 4)
    tj = TypedJaxArray(
        None, Type(st=(N,), et=Tensor(dims=(N,), name="X")),
    )
    assert tj.aa is None


@REQUIRES_TORCH
def test_typed_torch_tensor_aa_auto_populated(reset):
    """
    A `TypedTorchTensor` constructor should set `.aa` from the
    tensor's range, on either CPU or CUDA.
    """
    import torch
    from stile.torch._core import TypedTorchTensor
    N = dim("AaTtN", 4)
    arr = torch.tensor([-1.0, 0.5, 2.0, 3.5])
    tt = TypedTorchTensor(
        arr, Type(st=(N,), et=Tensor(dims=(N,), name="X")),
    )
    assert tt.aa is not None
    assert tt.aa.range() == (-1.0, 3.5)
