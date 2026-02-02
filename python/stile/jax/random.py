try:
    import jax
except ImportError:
    raise ImportError(
        "JAX support requires the jax extra: pip install stile[jax]"
    ) from None

from ..type import FullDim, Tensor, Type, dim_size
from ._core import TypedJaxArray


def normal(key : jax.Array, *shape : FullDim) -> TypedJaxArray:
    jax_shape = tuple(dim_size(d) for d in shape)
    arr = jax.random.normal(key, jax_shape)
    type = Type(
        dt=shape,
        et=Tensor(shape),
    )
    return TypedJaxArray(arr, type)
