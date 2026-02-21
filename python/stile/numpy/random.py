import numpy as np

from ..type import FullDim, Tensor, Type, dim_size
from ._core import TypedNumpyArray


def randn(*shape : FullDim) -> TypedNumpyArray:
    np_shape = tuple(dim_size(d) for d in shape)
    arr = np.random.randn(*np_shape)
    type = Type(
        st=shape,
        et=Tensor(shape),
    )
    return TypedNumpyArray(arr, type)
