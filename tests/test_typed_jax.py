import sys

import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjnp

from stile import expr_simplifies, reset_stile, dim

@pytest.fixture
def reset():
    yield
    reset_stile()


def test_simple_expression(reset):
    key = jax.random.PRNGKey(0)
    M, N = dim('M', 10), dim('N', 10)
    a = tjnp.random.normal(key, M, N)
    b = tjnp.TypedResult("2 * M N")
    for i in range(0, 10, 5):
        a_tile = a.slice(M, i, i + 5)
        a_scaled = a_tile * 2
        b.assign(a_scaled)


def test_basic_matmul(reset):
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    M, N, K = dim('M', 10), dim('N', 15), dim('K', 20)
    a = tjnp.random.normal(k1, M, N)
    b = tjnp.random.normal(k2, N, K)
    c = tjnp.TypedResult("(M N, N K -> M K)")

    for im in range(0, 10, 5):
        for ik in range(0, 20, 5):
            c_accum = 0
            for in_ in range(0, 15, 5):
                tile_a = a.slice(M, im, im + 5).slice(N, in_, in_ + 5)
                tile_b = b.slice(N, in_, in_ + 5).slice(K, ik, ik + 5)
                tile_c = tjnp.einsum(tile_a, tile_b, "M N, N K -> M K")
                c_accum = c_accum + tile_c
            assert isinstance(c_accum, tjnp.TypedJaxArray)
            c.assign(c_accum)


def test_exp(reset):
    key = jax.random.PRNGKey(0)
    M, N = dim('M', 10), dim('N', 10)
    a = tjnp.random.normal(key, M, N)
    c = tjnp.TypedResult("exp(M N)")

    a_exped = tjnp.exp(a)

    a_exped_div_a_exped = a_exped / a_exped
    a_exped_div_a_exped.assert_equivalent("1")

    c.assign(a_exped)
    assert jnp.allclose(
        c.arr,
        jnp.exp(a.arr),
    )

def test_numerically_stable_softmax(reset):
    key = jax.random.PRNGKey(0)
    N = dim('N', 8)
    x = tjnp.random.normal(key, N)

    # exp(x) / sum(exp(x))
    naive = tjnp.TypedResult("softmax[N](N)")
    x_max = x.max(N)
    assert expr_simplifies(x_max, "max[N](N)")
    
    x_stable = x - x_max.repeat(N)
    assert expr_simplifies(x_stable, "N - (max[N](N) -> N)")

    exp_x = tjnp.exp(x_stable)
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
    key = jax.random.PRNGKey(0)
    N = dim('N', 8)
    x = tjnp.random.normal(key, N)
    online = tjnp.TypedResult("softmax[N](N)")

    x_block1 = x.slice(N, 0, 4)
    m1 = x_block1.max(N)
    exp1 = tjnp.exp(x_block1 - m1.repeat(N[0:4]))
    l1 = exp1.sum(N)

    x_block2 = x.slice(N, 4, 8)
    m2 = x_block2.max(N)
    exp2 = tjnp.exp(x_block2 - m2.repeat(N[4:8]))
    l2 = exp2.sum(N)

    m_global = tjnp.maximum(m1, m2)

    l1_correction = tjnp.exp(m1 - m_global)
    l1_corrected = l1 * l1_correction

    l2_correction = tjnp.exp(m2 - m_global)
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

    softmax1 = tjnp.exp(x_block1 - m_global.repeat(N[0:4])) / l_global.repeat(N[0:4])
    softmax2 = tjnp.exp(x_block2 - m_global.repeat(N[4:8])) / l_global.repeat(N[4:8])
    online.assign(softmax1)
    online.assign(softmax2)


def softmax_jnp(x):
    ex = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return ex / jnp.sum(ex, axis=-1, keepdims=True)


def attention_jnp(q, k, v):
    qk = jnp.einsum('qd,nd->nq', q, k) / jnp.sqrt(q.shape[1])
    logits = softmax_jnp(qk)
    result = jnp.einsum('nq,nd->qd')
    return result


def test_flash_attention(reset):
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    dhead, qctx, nctx = dim('dhead', 16), dim('qctx', 32), dim('nctx', 128)

    Q = tjnp.random.normal(k1, qctx, dhead)
    K = tjnp.random.normal(k2, nctx, dhead)
    V = tjnp.random.normal(k3, nctx, dhead)

    L = tjnp.TypedResult("((softmax[nctx](qctx dhead, nctx dhead -> qctx nctx) / sqrt(16)), nctx dhead -> qctx dhead)")

    qctx_tile_size = 32
    nctx_tile_size = 32
    for iqctx in range(0, qctx.size, qctx_tile_size):
        running_max = -jnp.inf
        running_l = 0
        o = 0

        for ictx in range(0, nctx.size, nctx_tile_size):
            q_tile = Q.slice(qctx, iqctx, iqctx + qctx_tile_size)
            k_tile = K.slice(nctx, ictx, ictx + nctx_tile_size)

            qk_tile = tjnp.einsum(q_tile, k_tile, "qctx dhead, nctx dhead -> nctx qctx") / jnp.sqrt(dhead.size)
            tile_max = qk_tile.max(nctx)
            logits = tjnp.exp(qk_tile - tile_max.repeat(qk_tile.type.dt[0]))
            
            tile_l = logits.sum(nctx)
            new_max = tjnp.maximum(tile_max, running_max)
            new_l = tjnp.exp(running_max - new_max) * running_l + tjnp.exp(tile_max - new_max) * tile_l
            
            v_tile = V.slice(nctx, ictx, ictx + nctx_tile_size)
            v_proj = tjnp.einsum(logits, v_tile, "nctx qctx, nctx dhead -> qctx dhead")
            
            rescaled_old_o = (running_l * tjnp.exp(running_max - new_max)).repeat(dhead).rearrange(qctx, dhead) * o
            rescaled_v_proj = tjnp.exp(tile_max - new_max).repeat(dhead).rearrange(qctx, dhead) * v_proj

            o = (rescaled_old_o + rescaled_v_proj) / new_l.repeat(dhead).rearrange(qctx, dhead)
            running_l = new_l
            running_max = new_max
            o.assert_equivalent(
                "((softmax[nctx](qctx dhead, nctx dhead -> qctx nctx) / sqrt(16)), nctx dhead -> qctx dhead)",
                nctx[:(ictx + nctx_tile_size)],
            )

        assert isinstance(o, tjnp.TypedJaxArray)
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
