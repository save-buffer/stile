try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch support requires the torch extra: pip install stile[torch]"
    ) from None

from ..type import FullDim, Tensor, Type, dim_size
from ._core import TypedTorchTensor


def randn(*shape : FullDim, device : str = "cpu") -> TypedTorchTensor:
    torch_shape = tuple(dim_size(d) for d in shape)
    tensor = torch.randn(torch_shape, device=device)
    type = Type(
        st=shape,
        et=Tensor(shape),
    )
    return TypedTorchTensor(tensor, type)
