try:
    import triton
    import triton.language as tl
except ImportError:
    raise ImportError(
        "Triton support requires the triton extra: pip install stile[triton]"
    ) from None

import torch
import inspect
from functools import wraps

import stile.type as t
from ..type import *
from ..torch import TypedTorchTensor, TypedResult
from ..specification import parse_spec_into_type
from ..verification import verify_types_equivalent, verify_exprs_equivalent



def typed_jit(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):

        unwrapped_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, TypedTorchTensor):
                unwrapped_args.append(


            if isinstance(arg, TypedTorchTensor):
                unwrapped_args.append(arg.tensor)
            elif isinstance(arg, TypedResult):
                unwrapped_args.append(arg.tensor)
                typed_results.append(arg)
            else:
                unwrapped_args.append(arg)

        # Unwrap keyword arguments
        unwrapped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, TypedTorchTensor):
                unwrapped_kwargs[key] = value.tensor
            elif isinstance(value, TypedResult):
                unwrapped_kwargs[key] = value.tensor
                typed_results.append(value)
                result_positions.append(('kwarg', key))
            else:
                unwrapped_kwargs[key] = value

        # Call the underlying kernel
        result = fn(*unwrapped_args, **unwrapped_kwargs)

        return result

    return wrapper


class TypedKernelLauncher:
    """
    A helper for launching Triton kernels with typed tensors.
    """

    def __init__(self, kernel, grid):
        self.kernel = kernel
        self.grid = grid

    def __call__(self, *args, **kwargs):
        # Track inputs and outputs
        inputs = []
        outputs = []

        # Unwrap positional arguments
        unwrapped_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, TypedTorchTensor):
                unwrapped_args.append(arg.tensor)
                inputs.append((i, arg))
            elif isinstance(arg, TypedResult):
                unwrapped_args.append(arg.tensor)
                outputs.append((i, arg))
            else:
                unwrapped_args.append(arg)

        # Unwrap keyword arguments
        unwrapped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, TypedTorchTensor):
                unwrapped_kwargs[key] = value.tensor
                inputs.append((key, value))
            elif isinstance(value, TypedResult):
                unwrapped_kwargs[key] = value.tensor
                outputs.append((key, value))
            else:
                unwrapped_kwargs[key] = value

        # Launch the kernel
        self.kernel[self.grid](*unwrapped_args, **unwrapped_kwargs)


def launch(kernel, grid):
    """
    Create a typed kernel launcher.

    Usage:
        st.launch(my_kernel, grid)(x, output, n_elements, BLOCK_SIZE=256)

    Arguments should be:
    - TypedTorchTensor for inputs
    - TypedResult for outputs
    - Plain values for sizes, block sizes, etc.
    """
    return TypedKernelLauncher(kernel, grid)


def get_type(typed_tensor : TypedTorchTensor) -> Type:
    """Get the type from a TypedTorchTensor."""
    return typed_tensor.type


def wrap(tensor : torch.Tensor, type : Type) -> TypedTorchTensor:
    """Wrap a raw tensor with a type."""
    return TypedTorchTensor(tensor, type)

