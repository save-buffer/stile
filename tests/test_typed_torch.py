import sys

import torch
import pytest

import stile.torch as ttorch

from stile import expr_simplifies, reset_stile, dim


def test_simple_expression(reset):
    M, N = dim('M', 10), dim('N', 10)
    a = ttorch.random.randn(M, N)
    b = ttorch.TypedResult("2 * M N")
    for i in range(0, 10, 5):
        a_tile = a.slice(M, i, i + 5)
        a_scaled = a_tile * 2
        b.assign(a_scaled)


def test_basic_matmul(reset):
    M, N, K = dim('M', 10), dim('N', 15), dim('K', 20)
    a = ttorch.random.randn(M, N)
    b = ttorch.random.randn(N, K)
    c = ttorch.TypedResult("(M N, N K -> M K)")

    for im in range(0, 10, 5):
        for ik in range(0, 20, 5):
            c_accum = 0
            for in_ in range(0, 15, 5):
                tile_a = a.slice(M, im, im + 5).slice(N, in_, in_ + 5)
                tile_b = b.slice(N, in_, in_ + 5).slice(K, ik, ik + 5)
                tile_c = ttorch.einsum(tile_a, tile_b, "M N, N K -> M K")
                c_accum = c_accum + tile_c
            assert isinstance(c_accum, ttorch.TypedTorchTensor)
            c.assign(c_accum)


def test_abs_relu_minimum(reset):
    """
    `ttorch.abs/relu/minimum` lower to the same ETs as the matching
    spec keywords, so `assert_equivalent` is a structural identity.
    Numerical checks against torch's reference ops too.
    """
    M, N = dim('AbsTM', 6), dim('AbsTN', 4)
    a = ttorch.random.randn(M, N, name="A")
    b = ttorch.random.randn(M, N, name="B")

    abs_a = ttorch.abs(a)
    abs_a.assert_equivalent("abs(A:AbsTM AbsTN)")
    assert torch.allclose(abs_a.tensor, a.tensor.abs())

    relu_a = ttorch.relu(a)
    relu_a.assert_equivalent("relu(A:AbsTM AbsTN)")
    assert torch.allclose(relu_a.tensor, torch.maximum(a.tensor, torch.zeros_like(a.tensor)))

    min_ab = ttorch.minimum(a, b)
    min_ab.assert_equivalent("minimum(A:AbsTM AbsTN, B:AbsTM AbsTN)")
    assert torch.allclose(min_ab.tensor, torch.minimum(a.tensor, b.tensor))


def test_exp(reset):
    M, N = dim('M', 10), dim('N', 10)
    a = ttorch.random.randn(M, N)
    c = ttorch.TypedResult("exp(M N)")

    a_exped = ttorch.exp(a)

    a_exped_div_a_exped = a_exped / a_exped
    a_exped_div_a_exped.assert_equivalent("1")

    c.assign(a_exped)
    assert torch.allclose(
        c.tensor,
        torch.exp(a.tensor),
    )


def test_numerically_stable_softmax(reset):
    N = dim('N', 8)
    x = ttorch.random.randn(N, name="x")

    # exp(x) / sum(exp(x))
    naive = ttorch.TypedResult("softmax[N](x:N)")
    x_max = x.max(N)
    assert expr_simplifies(x_max, "max[N](x:N)")

    x_stable = x - x_max.repeat(N)
    assert expr_simplifies(x_stable, "x:N - (max[N](x:N) -> N)")

    exp_x = ttorch.exp(x_stable)
    assert expr_simplifies(exp_x, "exp(x:N - (max[N](x:N) -> N))")

    sum_exp_x = exp_x.sum(N).repeat(N)
    assert expr_simplifies(sum_exp_x, "(sum[N](exp(x:N - (max[N](x:N) -> N))) -> N)")

    softmax = exp_x / sum_exp_x
    # Literally what softmax is
    assert expr_simplifies(
        softmax,
        "(exp(x:N - (max[N](x:N) -> N))) / (sum[N](exp(x:N - (max[N](x:N) -> N))) -> N)",
    )
    # Convert exponent to quotient in numerator
    assert expr_simplifies(
        softmax,
        "(exp(x:N) / exp(max[N](x:N) -> N)) / (sum[N](exp(x:N - (max[N](x:N) -> N))) -> N)",
    )
    # Component exponent to quotient in denominator
    assert expr_simplifies(
        softmax,
        "(exp(x:N) / exp(max[N](x:N) -> N)) / (sum[N](exp(x:N) / exp(max[N](x:N) -> N)) -> N)",
    )
    # Pull repeat outside of the exp in the denominator
    assert expr_simplifies(
        softmax,
        "(exp(x:N) / exp(max[N](x:N) -> N)) / (sum[N](exp(x:N) / (exp(max[N](x:N)) -> N)) -> N)",
    )
    # Pull the denominator outside of the sum
    assert expr_simplifies(
        softmax,
        "(exp(x:N) / exp(max[N](x:N) -> N)) / (sum[N](exp(x:N)) / exp(max[N](x:N)) -> N)",
    )
    # Duplicate repeat in demoniator
    assert expr_simplifies(
        softmax,
        "(exp(x:N) / exp(max[N](x:N) -> N)) / ((sum[N](exp(x:N)) -> N) / (exp(max[N](x:N)) -> N))",
    )
    # Flip the denominator and turn it into a multiplication
    assert expr_simplifies(
        softmax,
        "(exp(x:N) / exp(max[N](x:N) -> N)) * ((exp(max[N](x:N)) -> N) / (sum[N](exp(x:N)) -> N))",
    )
    # Turn it into quotient of products
    assert expr_simplifies(
        softmax,
        "(exp(x:N) * (exp(max[N](x:N)) -> N)) / (exp(max[N](x:N) -> N) * (sum[N](exp(x:N)) -> N))",
    )
    # Associativity of multiplication in denominator
    assert expr_simplifies(
        softmax,
        "(exp(x:N) * (exp(max[N](x:N)) -> N)) / ((sum[N](exp(x:N)) -> N) * exp(max[N](x:N) -> N))",
    )
    # Break back into multiplication of fractions
    assert expr_simplifies(
        softmax,
        "(exp(x:N) / (sum[N](exp(x:N)) -> N)) * ((exp(max[N](x:N)) -> N) / (exp(max[N](x:N)) -> N))",
    )
    # Check that RHS fraction simplifies to 1
    assert expr_simplifies(
       softmax,
       "(exp(x:N) / (sum[N](exp(x:N)) -> N)) * 1",
    )

    naive.assign(softmax)


def test_online_softmax(reset):
    N = dim('N', 8)
    x = ttorch.random.randn(N, name="x")
    online = ttorch.TypedResult("softmax[N](x:N)")

    x_block1 = x.slice(N, 0, 4)
    m1 = x_block1.max(N)
    exp1 = ttorch.exp(x_block1 - m1.repeat(N[0:4]))
    l1 = exp1.sum(N)

    x_block2 = x.slice(N, 4, 8)
    m2 = x_block2.max(N)
    exp2 = ttorch.exp(x_block2 - m2.repeat(N[4:8]))
    l2 = exp2.sum(N)

    m_global = ttorch.maximum(m1, m2)

    l1_correction = ttorch.exp(m1 - m_global)
    l1_corrected = l1 * l1_correction

    l2_correction = ttorch.exp(m2 - m_global)
    l2_corrected = l2 * l2_correction
    l_global = l1_corrected + l2_corrected

    assert expr_simplifies(m_global, "max[N](x:N)")
    assert expr_simplifies(l1, "sum[N](exp(x:N[0:4] - (max[N](x:N[0:4]) -> N[0:4])))")
    assert expr_simplifies(l2, "sum[N](exp(x:N[4:8] - (max[N](x:N[4:8]) -> N[4:8])))")
    assert expr_simplifies(l1, "sum[N](exp(x:N[0:4])) / exp(max[N](x:N[0:4]))")
    assert expr_simplifies(l1_correction, "exp(max[N](x:N[0:4])) / exp(max[N](x:N))")
    assert expr_simplifies(l1_corrected, "(sum[N](exp(x:N[0:4])) / exp(max[N](x:N[0:4]))) * (exp(max[N](x:N[0:4])) / exp(max[N](x:N)))")
    assert expr_simplifies(l1_corrected, "(sum[N](exp(x:N[0:4])) * exp(max[N](x:N[0:4]))) / (exp(max[N](x:N[0:4])) * exp(max[N](x:N)))")
    assert expr_simplifies(l1_corrected, "(sum[N](exp(x:N[0:4])) * exp(max[N](x:N[0:4]))) / (exp(max[N](x:N)) * exp(max[N](x:N[0:4])))")
    assert expr_simplifies(
        l1_corrected,
        "(sum[N](exp(x:N[0:4])) / exp(max[N](x:N))) * (exp(max[N](x:N[0:4])) / exp(max[N](x:N[0:4])))",
    )
    assert expr_simplifies(
        l1_corrected,
        "(sum[N](exp(x:N[0:4])) / exp(max[N](x:N))) * 1",
    )
    assert expr_simplifies(
        l1_corrected,
        "sum[N](exp(x:N[0:4])) / exp(max[N](x:N))",
    )
    assert expr_simplifies(
        l2_corrected,
        "sum[N](exp(x:N[4:8])) / exp(max[N](x:N))",
    )
    assert expr_simplifies(
        l_global,
        "(sum[N](exp(x:N[0:4])) / exp(max[N](x:N))) + (sum[N](exp(x:N[4:8])) / exp(max[N](x:N)))",
    )
    assert expr_simplifies(
        l_global,
        "(sum[N](exp(x:N[0:4])) + sum[N](exp(x:N[4:8]))) / exp(max[N](x:N))",
    )
    assert expr_simplifies(
        l_global,
        "sum[N](exp(x:N)) / exp(max[N](x:N))",
    )
    assert expr_simplifies(
        l_global,
        "sum[N](exp(x:N) / (exp(max[N](x:N)) -> N))",
    )
    assert expr_simplifies(
        l_global,
        "sum[N](exp(x:N - (max[N](x:N) -> N)))",
    )

    softmax1 = ttorch.exp(x_block1 - m_global.repeat(N[0:4])) / l_global.repeat(N[0:4])
    softmax2 = ttorch.exp(x_block2 - m_global.repeat(N[4:8])) / l_global.repeat(N[4:8])
    online.assign(softmax1)
    online.assign(softmax2)


def softmax_torch(x, dim):
    ex = torch.exp(x - x.amax(dim=dim, keepdim=True))
    return ex / ex.sum(dim=dim, keepdim=True)


def attention_torch(q, k, v):
    qk = torch.einsum('qd,nd->qn', q, k) / (q.shape[1] ** 0.5)
    logits = softmax_torch(qk, dim=-1)
    return torch.einsum('qn,nd->qd', logits, v)


def test_flash_attention(reset):
    dhead, qctx, nctx = dim('dhead', 16), dim('qctx', 32), dim('nctx', 128)

    Q = ttorch.random.randn(qctx, dhead)
    K = ttorch.random.randn(nctx, dhead)
    V = ttorch.random.randn(nctx, dhead)

    L = ttorch.TypedResult("(softmax[nctx]((qctx dhead, nctx dhead -> qctx nctx) / sqrt(16)), nctx dhead -> qctx dhead)")

    qctx_tile_size = 32
    nctx_tile_size = 32
    for iqctx in range(0, qctx.size, qctx_tile_size):
        running_max = -float('inf')
        running_l = 0
        o = 0

        for ictx in range(0, nctx.size, nctx_tile_size):
            q_tile = Q.slice(qctx, iqctx, iqctx + qctx_tile_size)
            k_tile = K.slice(nctx, ictx, ictx + nctx_tile_size)

            qk_tile = ttorch.einsum(q_tile, k_tile, "qctx dhead, nctx dhead -> nctx qctx") / (dhead.size ** 0.5)
            tile_max = qk_tile.max(nctx)
            logits = ttorch.exp(qk_tile - tile_max.repeat(qk_tile.type.st[0]))

            tile_l = logits.sum(nctx)
            new_max = ttorch.maximum(tile_max, running_max)
            new_l = ttorch.exp(running_max - new_max) * running_l + ttorch.exp(tile_max - new_max) * tile_l

            v_tile = V.slice(nctx, ictx, ictx + nctx_tile_size)
            v_proj = ttorch.einsum(logits, v_tile, "nctx qctx, nctx dhead -> qctx dhead")

            rescaled_old_o = (running_l * ttorch.exp(running_max - new_max)).repeat(dhead).rearrange(qctx, dhead) * o
            rescaled_v_proj = ttorch.exp(tile_max - new_max).repeat(dhead).rearrange(qctx, dhead) * v_proj

            o = (rescaled_old_o + rescaled_v_proj) / new_l.repeat(dhead).rearrange(qctx, dhead)
            running_l = new_l
            running_max = new_max
            o.assert_equivalent(
                "(softmax[nctx]((qctx dhead, nctx dhead -> qctx nctx) / sqrt(16)), nctx dhead -> qctx dhead)",
                nctx[:(ictx + nctx_tile_size)],
            )

        assert isinstance(o, ttorch.TypedTorchTensor)
        L.assign(o)

    expected = attention_torch(Q.tensor, K.tensor, V.tensor)
    assert torch.allclose(L.tensor, expected, atol=1e-5), \
        "Flash Attention output does not numerically match reference attention"
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
