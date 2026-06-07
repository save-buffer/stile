from typing import cast

import numpy as np

from ..type import FullDim, Tensor, Type, dim_size
from ._core import TypedNumpyArray


def randn(*shape : FullDim, name : str | None = None) -> TypedNumpyArray:
    np_shape = tuple(dim_size(d) for d in shape)
    arr = cast("np.ndarray", np.random.randn(*np_shape))
    type = Type(
        st=shape,
        et=Tensor(shape, name=name),
    )
    return TypedNumpyArray(arr, type)
