"""
Runtime-indexed `scatter` and MoE (mixture-of-experts) tests. Scatter
is the dual of gather — both are opaque to the verifier, so as long
as kernel and spec reference the same `idx` tensor by name, the
structural match holds across loops and reductions.

MoE is the realistic motivating use case: each token is dispatched
to one expert (chosen by `expert_id`). The simplest spec form is
per-token: `output[t] = W[expert_id[t]] @ X[t]` — which is just a
`gather` of expert weights followed by an einsum. The kernel tile-
walks tokens and does the same per-tile via the existing fori_loop
invariant machinery.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax
from stile import dim, reset_stile
from stile.specification import parse_spec_into_type
from stile.verification import verify_exprs_equivalent


def test_scatter_same_idx_structural_equality(reset):
    """
    Two `scatter`s with identical source, dim, and idx tensor
    produce structurally identical expressions — opaque scatter
    matches by name.
    """
    M = dim("M", 8)
    N = dim("N", 32)
    D = dim("D", 4)
    X = tjax.random.normal(jax.random.PRNGKey(0), M, D, name="X")
    idx = tjax.runtime_index("idx", M, values_in=N)
    a = X.scatter(N, idx)
    b = X.scatter(N, idx)
    assert verify_exprs_equivalent(a.type.et, b.type.et)


def test_scatter_string_syntax(reset):
    """
    The spec parser recognizes `scatter[dim_in_dest](source, idx)`
    just like `gather[...]`, and produces the same `Scatter` ExprType
    a programmatic `.scatter(...)` would.
    """
    M = dim("M", 8)
    N = dim("N", 32)
    D = dim("D", 4)
    X = tjax.random.normal(jax.random.PRNGKey(0), M, D, name="X")
    idx = tjax.runtime_index("idx", M, values_in=N)
    programmatic = X.scatter(N, idx)
    spec = parse_spec_into_type(
        "scatter[N](X:M D, idx:M) -> N D"
    )
    assert verify_exprs_equivalent(programmatic.type.et, spec.et)


def test_scatter_gather_permutation_round_trip(reset):
    """
    When `perm` is declared a permutation,
    `scatter(gather(Y, perm), perm) = Y` and the dual identity hold.
    The verifier recognizes the wrapped scatter/gather, checks the
    `permutation` property on the inner index, and unwraps to `Y`
    directly.
    """
    N = dim("N", 16)
    D = dim("D", 4)
    Y = tjax.random.normal(jax.random.PRNGKey(0), N, D, name="Y")
    perm = tjax.runtime_index("perm", N, values_in=N, permutation=True)

    # gather then scatter
    forward = Y.gather(N, perm).scatter(N, perm)
    assert verify_exprs_equivalent(forward.type.et, Y.type.et)

    # scatter then gather (same permutation)
    backward = Y.scatter(N, perm).gather(N, perm)
    assert verify_exprs_equivalent(backward.type.et, Y.type.et)


def test_scatter_gather_non_permutation_does_not_collapse(reset):
    """
    Without the `permutation` property, the verifier doesn't apply
    the round-trip rewrite — the wrapped scatter/gather is *not*
    structurally equal to the inner source. Guards against the
    rewrite firing accidentally on opaque indices.
    """
    N = dim("N", 16)
    D = dim("D", 4)
    Y = tjax.random.normal(jax.random.PRNGKey(0), N, D, name="Y")
    idx = tjax.runtime_index("idx", N, values_in=N)  # no permutation flag

    forward = Y.gather(N, idx).scatter(N, idx)
    assert not verify_exprs_equivalent(forward.type.et, Y.type.et)


def test_moe_per_token_dispatch(reset):
    """
    The simplest MoE form: each token's output is the matmul of its
    expert's weight matrix with the token's input. Expressed as
    `gather[N_exp](W, expert_id)` followed by per-token einsum — no
    scatter required. Verified structurally and numerically.
    """
    n_tokens = dim("n_tokens", 16)
    d_in = dim("d_in", 4)
    d_out = dim("d_out", 8)
    n_experts = dim("n_experts", 4)

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    X = tjax.random.normal(k1, n_tokens, d_in, name="X")
    W = tjax.random.normal(k2, n_experts, d_in, d_out, name="W")
    eid_values = jax.random.randint(k3, (n_tokens.size,), 0, n_experts.size)
    expert_id = tjax.runtime_index(
        "expert_id", n_tokens, values_in=n_experts, arr=eid_values,
    )

    W_per_token = W.gather(n_experts, expert_id)
    output = tjax.einsum(
        W_per_token, X,
        "n_tokens d_in d_out, n_tokens d_in -> n_tokens d_out",
    )

    spec = parse_spec_into_type(
        "(gather[n_experts](W:n_experts d_in d_out, expert_id:n_tokens), "
        "X:n_tokens d_in -> n_tokens d_out) -> n_tokens d_out"
    )
    assert verify_exprs_equivalent(output.type.et, spec.et)

    expected = jnp.einsum("nd,nde->ne", X.arr, W.arr[eid_values])
    assert jnp.allclose(output.arr, expected, atol=1e-5)


def test_moe_sort_based_kernel(reset):
    """
    The realistic MoE kernel shape: sort tokens by expert (via a
    declared-permutation `π`), batch-matmul per expert in the sorted
    layout, then unsort via scatter. The verifier collapses the whole
    sort-compute-unsort to the per-token spec
    `Y[t] = W[expert_id[t]] @ X[t]` via three composable identities:
      - gather-of-gather hoist: `gather(W, gather(eid, π))`
        canonicalizes to `gather(gather(W, eid), π)`, so π appears as
        the outer indexing on every n_tokens-shaped factor.
      - gather-through-reduce: the einsum's reduce over `d_in` factors
        the common outer gather-by-π out, since `d_in` isn't in π's
        output dim. Result is `gather(per_token_spec, π)`.
      - scatter-gather permutation round-trip: the outer
        `scatter(_, π)` unwraps `gather(per_token_spec, π)` back to
        `per_token_spec`.
    None of these require `eid` to have any property (the partition
    structure is irrelevant when we use the sorted layout) — only `π`
    being a permutation is needed. Numerically verified.
    """
    n_tokens = dim("n_tokens", 16)
    d_in = dim("d_in", 4)
    d_out = dim("d_out", 8)
    n_experts = dim("n_experts", 4)

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    X = tjax.random.normal(k1, n_tokens, d_in, name="X")
    W = tjax.random.normal(k2, n_experts, d_in, d_out, name="W")
    # Concrete eid + matching argsort permutation so the kernel runs.
    eid_values = jax.random.randint(k3, (n_tokens.size,), 0, n_experts.size)
    pi_values = jnp.argsort(eid_values)
    expert_id = tjax.runtime_index(
        "expert_id", n_tokens, values_in=n_experts, arr=eid_values,
    )
    pi = tjax.runtime_index(
        "pi", n_tokens, values_in=n_tokens, permutation=True, arr=pi_values,
    )

    X_sorted = X.gather(n_tokens, pi)
    expert_id_sorted = expert_id.gather(n_tokens, pi)
    W_per_sorted = W.gather(n_experts, expert_id_sorted)
    Y_sorted = tjax.einsum(
        X_sorted, W_per_sorted,
        "n_tokens d_in, n_tokens d_in d_out -> n_tokens d_out",
    )
    Y = Y_sorted.scatter(n_tokens, pi)

    expected_type = parse_spec_into_type(
        "(gather[n_experts](W:n_experts d_in d_out, expert_id:n_tokens), "
        "X:n_tokens d_in -> n_tokens d_out) -> n_tokens d_out"
    )
    assert verify_exprs_equivalent(Y.type.et, expected_type.et)

    expected = jnp.einsum("nd,nde->ne", X.arr, W.arr[eid_values])
    assert jnp.allclose(Y.arr, expected, atol=1e-5)


def test_moe_tile_walked_via_invariant(reset):
    """
    Tile-walked MoE summed across all tokens and outputs, wired
    through `@tjax.jit`. Kernel processes `BN` tokens per iteration,
    per-iter gathering the matching expert weights and doing the
    einsum. Final scalar accumulator matches a single full-shape
    einsum spec. Numerically verified.
    """
    n_tokens = dim("n_tokens", 16)
    d_in = dim("d_in", 4)
    d_out = dim("d_out", 8)
    n_experts = dim("n_experts", 4)
    BN = 4

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    X = tjax.random.normal(k1, n_tokens, d_in, name="X")
    W = tjax.random.normal(k2, n_experts, d_in, d_out, name="W")
    eid_values = jax.random.randint(k3, (n_tokens.size,), 0, n_experts.size)
    expert_id = tjax.runtime_index(
        "expert_id", n_tokens, values_in=n_experts, arr=eid_values,
    )

    moe_expr = (
        "(gather[n_experts](W:n_experts d_in d_out, expert_id:n_tokens), "
        "X:n_tokens d_in -> n_tokens d_out)"
    )
    inv = f"sum[d_out](sum[n_tokens where n_tokens < {BN} * k]({moe_expr}))"

    @tjax.jit(spec=f"sum[d_out](sum[n_tokens]({moe_expr})) -> ")
    def tile_walked_moe(X, W, expert_id):
        def body(k, acc):
            tok_slice = X.slice(n_tokens, k * BN, (k + 1) * BN)
            eid_slice = expert_id.slice(n_tokens, k * BN, (k + 1) * BN)
            W_tile = W.gather(n_experts, eid_slice)
            out_tile = tjax.einsum(
                W_tile, tok_slice,
                "n_tokens d_in d_out, n_tokens d_in -> n_tokens d_out",
            )
            return acc + out_tile.sum(n_tokens).sum(d_out)
        return tjax.fori_loop(
            0, n_tokens.size // BN, body,
            init_val=0.0, invariant=inv,
        )

    result = tile_walked_moe(X=X, W=W, expert_id=expert_id)
    expected = jnp.einsum("nd,nde->ne", X.arr, W.arr[eid_values]).sum()
    assert jnp.allclose(result.arr, expected, atol=1e-5)
