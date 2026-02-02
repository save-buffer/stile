import sys

import numpy as np
import pytest

import stile.numpy as tnp

from stile import expr_simplifies, reset_stile, dim

@pytest.fixture
def reset():
    yield
    reset_stile()


def test_simple_expression(reset):
    M, N = dim('M', 10), dim('N', 10)
    a = tnp.random.randn(M, N)
    b = tnp.TypedResult("2 * M N")
    for i in range(0, 10, 5):
        a_tile = a.slice(M, i, i + 5)
        a_scaled = a_tile * 2
        b.assign(a_scaled)


def test_basic_matmul(reset):
    M, N, K = dim('M', 10), dim('N', 15), dim('K', 20)
    a = tnp.random.randn(M, N)
    b = tnp.random.randn(N, K)
    c = tnp.TypedResult("(M N, N K -> M K)")

    for im in range(0, 10, 5):
        for ik in range(0, 20, 5):
            c_accum = 0
            for in_ in range(0, 15, 5):
                tile_a = a.slice(M, im, im + 5).slice(N, in_, in_ + 5)
                tile_b = b.slice(N, in_, in_ + 5).slice(K, ik, ik + 5)
                tile_c = tnp.einsum(tile_a, tile_b, "M N, N K -> M K")
                c_accum = c_accum + tile_c
            assert isinstance(c_accum, tnp.TypedNumpyArray)
            c.assign(c_accum)


def test_exp(reset):
    M, N = dim('M', 10), dim('N', 10)
    a = tnp.random.randn(M, N)
    c = tnp.TypedResult("exp(M N)")
    
    a_exped = tnp.exp(a)

    a_exped_div_a_exped = a_exped / a_exped
    a_exped_div_a_exped.assert_equivalent("1")

    c.assign(a_exped)
    np.testing.assert_allclose(
        c.arr,
        np.exp(a.arr),
    )

def test_numerically_stable_softmax(reset):
    N = dim('N', 8)
    x = tnp.random.randn(N)

    # exp(x) / sum(exp(x))
    naive = tnp.TypedResult("softmax[N](N)")
    x_max = x.max(N)
    assert expr_simplifies(x_max, "max[N](N)")
    
    x_stable = x - x_max.repeat(N)
    assert expr_simplifies(x_stable, "N - (max[N](N) -> N)")

    exp_x = tnp.exp(x_stable)
    assert expr_simplifies(exp_x, "exp(N - (max[N](N) -> N))")

    sum_exp_x = exp_x.sum(N).repeat(N)
    assert expr_simplifies(sum_exp_x, "(sum[N](exp(N - (max[N](N) -> N))) -> N)")

    softmax = exp_x / sum_exp_x
    # Literally what softmax is
    assert expr_simplifies(
        softmax,
        "(exp(N - (max[N](N) -> N))) / (sum[N](exp(N - (max[N](N) -> N))) -> N)",
    )
    # Convert exponent to quotient in numerator
    assert expr_simplifies(
        softmax,
        "(exp(N) / exp(max[N](N) -> N)) / (sum[N](exp(N - (max[N](N) -> N))) -> N)",
    )
    # Component exponent to quotient in denominator
    assert expr_simplifies(
        softmax,
        "(exp(N) / exp(max[N](N) -> N)) / (sum[N](exp(N) / exp(max[N](N) -> N)) -> N)",
    )
    # Pull repeat outside of the exp in the denominator
    assert expr_simplifies(
        softmax,
        "(exp(N) / exp(max[N](N) -> N)) / (sum[N](exp(N) / (exp(max[N](N)) -> N)) -> N)",
    )
    # Pull the denominator outside of the sum
    assert expr_simplifies(
        softmax,
        "(exp(N) / exp(max[N](N) -> N)) / (sum[N](exp(N)) / exp(max[N](N)) -> N)",
    )
    # Duplicate repeat in demoniator
    assert expr_simplifies(
        softmax,
        "(exp(N) / exp(max[N](N) -> N)) / ((sum[N](exp(N)) -> N) / (exp(max[N](N)) -> N))",
    )
    # Flip the denominator and turn it into a multiplication
    assert expr_simplifies(
        softmax,
        "(exp(N) / exp(max[N](N) -> N)) * ((exp(max[N](N)) -> N) / (sum[N](exp(N)) -> N))",
    )
    # Turn it into quotient of products
    assert expr_simplifies(
        softmax,
        "(exp(N) * (exp(max[N](N)) -> N)) / (exp(max[N](N) -> N) * (sum[N](exp(N)) -> N))",
    )
    # Associativity of multiplication in denominator
    assert expr_simplifies(
        softmax,
        "(exp(N) * (exp(max[N](N)) -> N)) / ((sum[N](exp(N)) -> N) * exp(max[N](N) -> N))",
    )
    # Break back into multiplication of fractions
    assert expr_simplifies(
        softmax,
        "(exp(N) / (sum[N](exp(N)) -> N)) * ((exp(max[N](N)) -> N) / (exp(max[N](N)) -> N))",
    )
    # Check that RHS fraction simplifies to 1
    assert expr_simplifies(
       softmax,
       "(exp(N) / (sum[N](exp(N)) -> N)) * 1",
    )

    naive.assign(softmax)

def test_online_softmax(reset):
    N = dim('N', 8)
    x = tnp.random.randn(N)
    online = tnp.TypedResult("softmax[N](N)")

    x_block1 = x.slice(N, 0, 4)
    m1 = x_block1.max(N)
    exp1 = tnp.exp(x_block1 - m1.repeat(N[0:4]))
    l1 = exp1.sum(N)

    x_block2 = x.slice(N, 4, 8)
    m2 = x_block2.max(N)
    exp2 = tnp.exp(x_block2 - m2.repeat(N[4:8]))
    l2 = exp2.sum(N)

    m_global = tnp.maximum(m1, m2)

    l1_correction = tnp.exp(m1 - m_global)
    l1_corrected = l1 * l1_correction

    l2_correction = tnp.exp(m2 - m_global)
    l2_corrected = l2 * l2_correction
    l_global = l1_corrected + l2_corrected

    assert expr_simplifies(m_global, "max[N](N)")
    assert expr_simplifies(l1, "sum[N](exp(N[0:4] - (max[N](N[0:4]) -> N[0:4])))")
    assert expr_simplifies(l2, "sum[N](exp(N[4:8] - (max[N](N[4:8]) -> N[4:8])))")
    assert expr_simplifies(l1, "sum[N](exp(N[0:4])) / exp(max[N](N[0:4]))")
    assert expr_simplifies(l1_correction, "exp(max[N](N[0:4])) / exp(max[N](N))")
    assert expr_simplifies(l1_corrected, "(sum[N](exp(N[0:4])) / exp(max[N](N[0:4]))) * (exp(max[N](N[0:4])) / exp(max[N](N)))")
    assert expr_simplifies(l1_corrected, "(sum[N](exp(N[0:4])) * exp(max[N](N[0:4]))) / (exp(max[N](N[0:4])) * exp(max[N](N)))")
    assert expr_simplifies(l1_corrected, "(sum[N](exp(N[0:4])) * exp(max[N](N[0:4]))) / (exp(max[N](N)) * exp(max[N](N[0:4])))")
    assert expr_simplifies(
        l1_corrected,
        "(sum[N](exp(N[0:4])) / exp(max[N](N))) * (exp(max[N](N[0:4])) / exp(max[N](N[0:4])))",
    )
    assert expr_simplifies(
        l1_corrected,
        "(sum[N](exp(N[0:4])) / exp(max[N](N))) * 1",
    )
    assert expr_simplifies(
        l1_corrected,
        "sum[N](exp(N[0:4])) / exp(max[N](N))",
    )
    assert expr_simplifies(
        l2_corrected,
        "sum[N](exp(N[4:8])) / exp(max[N](N))",
    )
    assert expr_simplifies(
        l_global,
        "(sum[N](exp(N[0:4])) / exp(max[N](N))) + (sum[N](exp(N[4:8])) / exp(max[N](N)))",
    )
    assert expr_simplifies(
        l_global,
        "(sum[N](exp(N[0:4])) + sum[N](exp(N[4:8]))) / exp(max[N](N))",
    )
    assert expr_simplifies(
        l_global,
        "sum[N](exp(N)) / exp(max[N](N))",
    )
    assert expr_simplifies(
        l_global,
        "sum[N](exp(N) / (exp(max[N](N)) -> N))",
    )
    assert expr_simplifies(
        l_global,
        "sum[N](exp(N - (max[N](N) -> N)))",
    )

    softmax1 = tnp.exp(x_block1 - m_global.repeat(N[0:4])) / l_global.repeat(N[0:4])
    softmax2 = tnp.exp(x_block2 - m_global.repeat(N[4:8])) / l_global.repeat(N[4:8])
    online.assign(softmax1)
    online.assign(softmax2)


def softmax_np(x):
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / np.sum(ex, axis=-1, keepdims=True)


def attention_np(q, k, v):
    qk = np.einsum('qd,nd->nq', q, k) / np.sqrt(q.shape[1])
    logits = softmax_np(qk)
    result = np.einsum('nq,nd->qd')
    return result


def test_flash_attention(reset):
    dhead, qctx, nctx = dim('dhead', 16), dim('qctx', 32), dim('nctx', 128)

    Q = tnp.random.randn(qctx, dhead)
    K = tnp.random.randn(nctx, dhead)
    V = tnp.random.randn(nctx, dhead)

    L = tnp.TypedResult("((softmax[nctx](qctx dhead, nctx dhead -> qctx nctx) / sqrt(16)), nctx dhead -> qctx dhead)")

    qctx_tile_size = 32
    nctx_tile_size = 32
    for iqctx in range(0, qctx.size, qctx_tile_size):
        running_max = -np.inf
        running_l = 0
        o = 0

        for ictx in range(0, nctx.size, nctx_tile_size):
            q_tile = Q.slice(qctx, iqctx, iqctx + qctx_tile_size)
            k_tile = K.slice(nctx, ictx, ictx + nctx_tile_size)
            
            qk_tile = tnp.einsum(q_tile, k_tile, "qctx dhead, nctx dhead -> nctx qctx") / np.sqrt(dhead.size)
            tile_max = qk_tile.max(nctx)
            logits = tnp.exp(qk_tile - tile_max.repeat(qk_tile.type.dt[0]))
            
            tile_l = logits.sum(nctx)
            new_max = tnp.maximum(tile_max, running_max)
            new_l = tnp.exp(running_max - new_max) * running_l + tnp.exp(tile_max - new_max) * tile_l
            
            v_tile = V.slice(nctx, ictx, ictx + nctx_tile_size)
            v_proj = tnp.einsum(logits, v_tile, "nctx qctx, nctx dhead -> qctx dhead")
            
            rescaled_old_o = (running_l * tnp.exp(running_max - new_max)).repeat(dhead).rearrange(qctx, dhead) * o
            rescaled_v_proj = tnp.exp(tile_max - new_max).repeat(dhead).rearrange(qctx, dhead) * v_proj

            o = (rescaled_old_o + rescaled_v_proj) / new_l.repeat(dhead).rearrange(qctx, dhead)
            running_l = new_l
            running_max = new_max
            o.assert_equivalent(
                "((softmax[nctx](qctx dhead, nctx dhead -> qctx nctx) / sqrt(16)), nctx dhead -> qctx dhead)",
                nctx[:(ictx + nctx_tile_size)],
            )

        assert isinstance(o, tnp.TypedNumpyArray)
        L.assign(o)
    print("Formally verified Flash Attention passed!")


tests = [
    test_simple_expression,
    test_basic_matmul,
    test_exp,
    test_numerically_stable_softmax,
    test_online_softmax,
    test_flash_attention,
]

if __name__ == '__main__':
    for test in tests:
        print("Running", test)
        sys.stdout.flush()
        reset_stile()
        test(None)
