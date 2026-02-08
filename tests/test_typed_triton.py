"""
Triton integration tests.

These tests require Linux (Triton only has Linux wheels).
Run with: TRITON_INTERPRET=1 python tests/test_typed_triton.py
"""
import os
import sys
import platform

# Enable interpreter mode for testing without GPU
os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl

import stile.torch as ttorch
import stile.triton as ttl

import pytest
from stile import dim, reset_stile
from stile.type import Type, Tensor, FullDim


@pytest.fixture
def reset():
    yield
    reset_stile()


@ttl.typed_jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE : tl.constexpr,
):
    """Vector addition kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def test_simple_add(reset):
    """Test basic typed vector addition."""
    N = dim('N', 1024)

    # Create typed tensors - inputs are TypedTorchTensor, outputs are TypedResult
    x = tpt.random.randn(N)
    y = tpt.random.randn(N)
    output = tpt.TypedResult("N")

    # Launch kernel with typed tensors using st.launch
    n_elements = N.size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    st.launch(add_kernel, grid)(
        x, y, output,
        n_elements,
        BLOCK_SIZE=256,
    )

    # Verify numerical correctness
    expected = x.tensor + y.tensor
    assert torch.allclose(output.tensor, expected)


@triton.jit
def exp_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE : tl.constexpr,
):
    """Exponential kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.exp(x)
    tl.store(output_ptr + offsets, output, mask=mask)


def test_exp(reset):
    """Test exponential operation with typed tensors."""
    N = dim('N', 512)

    x = tpt.random.randn(N)
    output = tpt.TypedResult("exp(N)")

    n_elements = N.size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    st.launch(exp_kernel, grid)(
        x,
        output,
        n_elements,
        BLOCK_SIZE=128,
    )

    expected = torch.exp(x.tensor)
    assert torch.allclose(output.tensor, expected)


@triton.jit
def softmax_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE : tl.constexpr,
):
    """
    Simple softmax kernel (processes entire vector in one block).
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

    x_max = tl.max(x, axis=0)
    x_stable = x - x_max
    exp_x = tl.exp(x_stable)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp

    tl.store(output_ptr + offsets, softmax, mask=mask)


def test_softmax(reset):
    """Test softmax kernel with typed tensors."""
    N = dim('N', 64)

    x = tpt.random.randn(N)
    output = tpt.TypedResult("softmax[N](N)")

    # Use st.unwrap for direct kernel call
    softmax_kernel[(1,)](
        st.unwrap(x),
        st.unwrap(output),
        N.size,
        BLOCK_SIZE=64,
    )

    expected = torch.softmax(x.tensor, dim=0)
    assert torch.allclose(output.tensor, expected, atol=1e-5)


tests = [
    test_simple_add,
    test_exp,
    test_softmax,
]

if __name__ == '__main__':
    if platform.system() != "Linux":
        print("Skipping Triton tests: only supported on Linux")
        sys.exit(0)

    for test in tests:
        print("Running", test.__name__)
        sys.stdout.flush()
        reset_stile()
        test(None)
    print("All Triton tests passed!")
