import numpy as np

from typed_numpy import (
    Typed,
    TypedResult,
    FullDim,
    einsum,
)

# Create untyped numpy arrays
a = np.random.randn(10, 10)
b = np.random.randn(10, 10)

# Declaring dimensions. You shouldn't create duplicate dimensions.
M, N, K = FullDim('M', 10), FullDim('N', 10), FullDim('K', 10)

# Wrap your numpy arrays inside of the Typed class, and specify their dimensions.
# This is the type system's entry point into its knowledge about your program.
a = Typed(a, M, N)
b = Typed(b, N, K)

# Slice things! a_slice and b_slice are sliced as if we did a[0:5, 0:5] and b[0:5, 0:5]
a_sliced = a.slice(M, 0, 5).slice(N, 0, 5)
b_sliced = b.slice(N, 0, 5).slice(K, 0, 5)

# The Typed object has a shape and a type. The type displayed by the `type` accessor is
# the _DimType_. There is also an _ExprType_, which we will return to later. 
print("Shape:", a_sliced.shape)
print("Type: ", a_sliced.type)

# Notice now that we can only perform operations on things with the same type:
try:
    a_sliced * b_sliced
except ValueError as e:
    pass

# We can actually mix dimensions using einsum! Internally this
# `repeat`s the dimensions (which is how you introduce new dimensions)
# and `reduce`s along the reduction dimensions of the einsum.
# Notice that the returned tile retains the slicing information from the input tensors.
c_tile0 = einsum(a_sliced, b_sliced, "M N, N K -> M K")

print("Shape:    ", c_tile0.shape)
print("DimType:  ", c_tile0.type)
print("ExprType: ", c_tile0.expr_type)

# Specifications: You can create a TypedResult using this specification string. You can only assign
# to tiles of the TypedResult if the ExprTypes match! Tensors are defined implicitly as sequences of dimensions.

# This requires an input tensor of shape (M K) and that it's multiplied by 2
TypedResult(spec="2 * M K")
# This requires that you perform a matmul, and then multiply by 2
TypedResult(spec="(M N, N K -> M K) * 2")
# This requires that you perform an einsum, but multiplying the tensors by 2 first.
TypedResult(spec="(2 * M N, 2 * N K -> M K)")

# Below we'll demonstrate a verified tiled matmul! Please note that einsums MUST be surrounded by parens (for now)
# Here we create a TypedResult c. Note that c infers its shape from the global dimension registry.
c = TypedResult("6 * (M N, N K -> M K)")

tile_size = 5
# We do our classic triple for loop
for im in range(0, 10, tile_size):
    for ik in range(0, 10, tile_size):
        # We start with our accumulator as None. 
        c_accum = None
        for in_ in range(0, 10, tile_size):
            # Slice the inputs and multiply by a constant.
            tile_a = a.slice(M, im, im + tile_size).slice(N, in_, in_ + tile_size)
            tile_b = b.slice(N, in_, in_ + tile_size).slice(K, ik, ik + tile_size)
            # Multiply the tiles by constants BEFORE the einsum. Notice that the type checker
            # will fold the constants, and guarantee that the result is what we expected, 
            # the matmul multiplied by 6!
            tile_a_scaled = tile_a * 3
            tile_b_scaled = tile_b * 2
            # Perform our tile-sized matmul
            tile_c = einsum(tile_a_scaled, tile_b_scaled, "M N, N K -> M K")
            # Accumulate the result
            c_accum = c_accum + tile_c if c_accum is not None else tile_c
        # Write it out to the result tensor! Notice that if we'd done something NOT according
        # to the specification, we would not be able to assign here. For instance, if our dimensions
        # were mismatching, or we failed to reduce the entire dimension, or if we accidentally multiplied
        # by 2. 
        assert c_accum is not None # Make typechecker happy
        c.assign(c_accum)

# Now we can get our result from c.arr
actual = c.arr
expected = 6 * (a.arr @ b.arr)
np.testing.assert_allclose(
    expected,
    actual,
)
print("The formally verified tiled matmul has passed!")
