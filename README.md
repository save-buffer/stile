# Stile

This is an experiment at adding a type system to Numpy for the purposes of writing tiled kernels. 
Tiled kernels are notoriously difficult to write, in no small part due to the fact that the inputs
and outputs to kernels are large piles of numbers. Debugging is difficult, and it's easy to multiply 
or add things that really shouldn't be added. This is an attempt to remedy that by augmenting tensors
with _logical- information about their dimensions, sizes, and current slicing. This should make kernel-writing
dramatically faster and easier than current state of the art. 

Numpy as the backing store was for convenience of developing this. For production scenarios, this
type system can and should be ported to Python DSLs for kernel-writing like Triton and Pallas. 

This type system really consists of two kinds of type, which are tracked automatically without manual
type hints:
- ShapeType: The type of a given tensor in terms of its dimensions, and how it's sliced up
- ExprType: The tree of operations performed up till this point

The rough rules that the type system enforces are:
- You may only perform binary operations on tensors that have corresponding slices along all of the same
logical dimensions. 
- You must reduce along an entire dimension before writing it to the result
- The resulting operations match the specification.

You can then specify your expected result using a lightweight specification language. For instance, 
if you want to compute `2 * A @ B` (in Numpy parlance), you would specify the string `2 * (M N, N K -> M K)`.
Then, after you write your tiled kernel, when you output the result, type checking will fail if the 
type system is unable to prove that your expression that you've built up is equivalent to the specification.
The expression simplifier is implemented as an Egraph using the wonderful Egglog library, and is powerful
enough to prove a tiled Flash Attention kernel!

## Example
```python
from stile import dim

import style.numpy as tnp # tnp is Typed Numpy

M, N, K = dim('M', 10), dim('N', 10), dim('K', 10)
a = tnp.randn(M, N)
b = tnp.randn(N, K)
c = tnp.TypedResult("6 * (M N, N K -> M K)")

for im in range(0, 10, tile_size):
    for ik in range(0, 10, tile_size):
        c_accum = 0
        for in_ in range(0, 10, tile_size):
            # Slice the inputs
            tile_a = a.slice(M, im, im + tile_size).slice(N, in_, in_ + tile_size)
            tile_b = b.slice(N, in_, in_ + tile_size).slice(K, ik, ik + tile_size)
            # Multiply the tiles by constants BEFORE the einsum. Notice that the type checker
            # will fold the constants, and guarantee that the result is what we expected, 
            # the matmul multiplied by 6!
            tile_a_scaled = tile_a * 3
            tile_b_scaled = tile_b * 2
            # Perform a single tile's worth of matmul
            tile_c = tnp.einsum(tile_a_scaled, tile_b_scaled, "M N, N K -> M K")
            # Accumulate the result
            c_accum = c_accum + tile_c
        # Write it out to the result tensor! Notice that if we'd done something NOT according
        # to the specification, we would not be able to assign here. For instance, if our dimensions
        # were mismatching, or we failed to reduce the entire dimension, or if we accidentally multiplied
        # by 2. 
        c.assign(c_accum)
```

While writing the demo above, I originally had the spec be `2 * (M N, N K -> M K)`, and had foolishly
multiplied both `tile_a` AND `tile_b` by `2`. Obviously this is incorrect, and the type system caught
the mistake!

## Performance
The most complicated thing that this system has been tested on is a simple Flash Attention, which
it can successfully validate (see `tests/test_stile.py::test_flash_attention`). The size of 
the expression that we prove grows with the number of _tiles_, meaning a context length of 512 with
tiles of size 32 will take the same amount of time to prove as a context length of 16,384 with a
tile size of 1024. In release mode on my Apple M3 Pro 18GB, the time to prove Flash Attention is given
below as a function of context length.

| Context Length | Number of Tiles | Time   |
|----------------|-----------------|--------|
| 128            | 4               | 1.18s  |
| 256            | 8               | 3.52s  |
| 512            | 16              | 67.53s |

