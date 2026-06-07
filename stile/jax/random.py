try:
    import jax
except ImportError:
    raise ImportError(
        "JAX support requires the jax extra: pip install stile[jax]"
    ) from None

from typing import cast

from ..type import FullDim, Tensor, Type, dim_size
from ._core import TypedJaxArray


def normal(
    key : jax.Array,
    *shape : FullDim,
    name : str | None = None,
) -> TypedJaxArray:
    """
    Sample a typed array with stile dim signature `shape`. `name`, if
    given, becomes the tensor's leaf identity — pass it whenever the
    spec uses a `label:dims` reference for this input. Unspecified gets
    a fresh auto-name.
    """
    jax_shape = cast("tuple[int, ...]", tuple(dim_size(d) for d in shape))
    arr = jax.random.normal(key, jax_shape)
    type = Type(
        st=shape,
        et=Tensor(shape, name=name),
    )
    return TypedJaxArray(arr, type)
