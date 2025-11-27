import numpy as np

from typed_numpy import (
    Typed,
    TypedResult,
    FullDim,
    einsum,
    exp,
    max,
    reset_typed_numpy,
)



def test_simple_expression():
    M, N = FullDim('M', 10), FullDim('N', 10)
    a = Typed(np.random.randn(10, 10), M, N)
    b = TypedResult("2 * M N")
    for i in range(0, 10, 5):
        a_tile = a.slice(M, i, i + 5)
        a_scaled = a_tile * 2
        b.assign(a_scaled)


def test_basic_matmul():
    M, N, K = FullDim('M', 10), FullDim('N', 15), FullDim('K', 20)
    a = Typed(np.random.randn(10, 15), M, N)
    b = Typed(np.random.randn(15, 20), N, K)
    c = TypedResult("(M N, N K -> M K)")

    for im in range(0, 10, 5):
        for ik in range(0, 20, 5):
            c_accum = None
            for in_ in range(0, 15, 5):
                tile_a = a.slice(M, im, im + 5).slice(N, in_, in_ + 5)
                tile_b = b.slice(N, in_, in_ + 5).slice(K, ik, ik + 5)
                tile_c = einsum(tile_a, tile_b, "M N, N K -> M K")
                c_accum = c_accum + tile_c if c_accum is not None else tile_c
            assert c_accum is not None
            c.assign(c_accum)

tests = [
    test_simple_expression,
    test_basic_matmul,
]

if __name__ == '__main__':
    for test in tests:
        reset_typed_numpy()
        test()
