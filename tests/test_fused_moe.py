"""
Fused MoE — the realistic kernel shape that exploits per-expert
batching: load `W[g]` once per expert, apply to all of expert `g`'s
contiguous block of (sorted) tokens, scatter back. The verifier
proves this kernel equals the per-token spec via the block-constant
rewrite: inside a slice bounded by `(offsets[g], offsets[g+1])`, a
`gather(W, eid_sorted)` collapses to `W[g]` because the paired index
`eid_sorted` holds value `g` everywhere on that block.
"""
import jax
import pytest

import stile.jax as tjax
from stile import dim, reset_stile, SymbolicInt, Type
from stile.indexing import to_affine
from stile.specification import parse_spec_into_type
from stile.type import ParametricReduce
from stile.verification import verify_exprs_equivalent


@pytest.fixture
def reset():
    yield
    reset_stile()


def test_fused_moe_single_block(reset):
    """One per-expert iteration of fused MoE versus the per-token
    spec sliced to that block. The kernel uses `W[g]` (singleton
    slice + reduce); the spec uses `gather(W, eid_sorted)`. Inside
    the block, the gather collapses to `W[g]` via the
    block-constant rewrite — both forms canonicalize identically."""
    n_tokens = dim("n_tokens", 16)
    d_in = dim("d_in", 4)
    d_out = dim("d_out", 8)
    n_experts = dim("n_experts", 4)
    n_offsets = dim("n_offsets", 5)  # n_experts + 1

    k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
    X_sorted = tjax.random.normal(k1, n_tokens, d_in, name="X_sorted")
    W = tjax.random.normal(k2, n_experts, d_in, d_out, name="W")

    offsets = tjax.runtime_index("offsets", n_offsets, values_in=n_tokens)
    eid_sorted = tjax.runtime_index(
        "eid_sorted", n_tokens, values_in=n_experts,
        block_sorted_paired_with="offsets",
    )

    # Symbolic block index — the rewrite handles `g` left free.
    g = SymbolicInt("g")

    # Kernel side: per-expert matmul. Slice `X_sorted` to the block
    # owned by expert `g`, pick `W[g]` (singleton via slice+sum),
    # matmul.
    W_g = W.slice(n_experts, g, to_affine(g) + 1).sum(n_experts)
    X_block_kernel = X_sorted.block_at(n_tokens, offsets, g)
    Y_block_kernel = tjax.einsum(
        X_block_kernel, W_g,
        "n_tokens d_in, d_in d_out -> n_tokens d_out",
    )
    total_kernel = Y_block_kernel.sum(n_tokens).sum(d_out)

    # Spec side: per-token form, restricted to the same block.
    spec_full = tjax.einsum(
        X_sorted, W.gather(n_experts, eid_sorted),
        "n_tokens d_in, n_tokens d_in d_out -> n_tokens d_out",
    )
    Y_block_spec = spec_full.block_at(n_tokens, offsets, g)
    total_spec = Y_block_spec.sum(n_tokens).sum(d_out)

    assert verify_exprs_equivalent(total_kernel.type.et, total_spec.type.et)


def test_fused_moe_typed_result(reset):
    """The real fused MoE kernel verified end-to-end via TypedResult.
    Each fori_loop iter computes the per-expert block matmul and
    assigns it to the output buffer via `L.assign(Y_block)`. The
    assigned tile carries `Sliced` block bounds on its type — `assign`
    wraps both the tile and the spec in `Reduce(Sliced(...), sum, ...)`
    so the block-constant rewrite fires (collapsing the spec's
    `gather(W, eid_sorted)` to `W[g]` inside the block) and the per-
    tile ETs match. After the loop, `done()` verifies the assigned
    blocks tile the full output dim via the boundary declarations.
    Both halves — per-tile correctness and full-dim coverage — verify
    the kernel against the per-token spec."""
    n_tokens = dim("n_tokens", 16)
    d_in = dim("d_in", 4)
    d_out = dim("d_out", 8)
    n_experts = dim("n_experts", 4)
    n_offsets = dim("n_offsets", 5)

    k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
    X_sorted = tjax.random.normal(k1, n_tokens, d_in, name="X_sorted")
    W = tjax.random.normal(k2, n_experts, d_in, d_out, name="W")

    offsets = tjax.runtime_index(
        "offsets", n_offsets, values_in=n_tokens,
        boundary_values={0: 0, n_experts.size: n_tokens.size},
    )
    tjax.runtime_index(
        "eid_sorted", n_tokens, values_in=n_experts,
        block_sorted_paired_with="offsets",
    )

    L = tjax.TypedResult(
        "(gather[n_experts](W:n_experts d_in d_out, eid_sorted:n_tokens), "
        "X_sorted:n_tokens d_in -> n_tokens d_out) -> n_tokens d_out"
    )

    def body(g, _):
        X_block = X_sorted.block_at(n_tokens, offsets, g)
        W_g = W.slice(n_experts, g, to_affine(g) + 1).sum(n_experts)
        Y_block = tjax.einsum(
            X_block, W_g,
            "n_tokens d_in, d_in d_out -> n_tokens d_out",
        )
        L.assign(Y_block)
        return None

    tjax.fori_loop(0, n_experts.size, body, None)
    L.done()


def test_fused_moe_write_block_carry(reset):
    """The carry-pattern version of fused MoE: the body threads a
    `TypedResult` as the `fori_loop` carry. Each iter computes the
    per-expert block matmul and writes it via `output.write_block(tile)`
    — which delegates to `assign` for per-tile verification (block-
    constant rewrite firing inside the slice) and records the bounds
    for coverage. After the loop, `done()` verifies the per-block
    writes tile the full output. Numerically compares the carry's
    populated array against the spec evaluated end-to-end on the same
    inputs.
    """
    import jax.numpy as jnp

    n_tokens = dim("n_tokens", 16)
    d_in = dim("d_in", 4)
    d_out = dim("d_out", 8)
    n_experts = dim("n_experts", 4)
    n_offsets = dim("n_offsets", 5)

    k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
    X_sorted = tjax.random.normal(k1, n_tokens, d_in, name="X_sorted")
    W = tjax.random.normal(k2, n_experts, d_in, d_out, name="W")

    # Concrete offsets + eid_sorted so the kernel can actually run end-
    # to-end. eid_sorted is non-decreasing (block-sorted) and offsets
    # are its block boundaries.
    eid_values = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    offsets_values = jnp.array([0, 4, 8, 12, 16])

    offsets = tjax.runtime_index(
        "offsets", n_offsets, values_in=n_tokens,
        boundary_values={0: 0, n_experts.size: n_tokens.size},
        arr=offsets_values,
    )
    eid_sorted = tjax.runtime_index(
        "eid_sorted", n_tokens, values_in=n_experts,
        block_sorted_paired_with="offsets",
        arr=eid_values,
    )

    L = tjax.TypedResult(
        "(gather[n_experts](W:n_experts d_in d_out, eid_sorted:n_tokens), "
        "X_sorted:n_tokens d_in -> n_tokens d_out) -> n_tokens d_out"
    )

    def body(g, output):
        X_block = X_sorted.block_at(n_tokens, offsets, g)
        W_g = W.slice(n_experts, g, to_affine(g) + 1).sum(n_experts)
        Y_block = tjax.einsum(
            X_block, W_g,
            "n_tokens d_in, d_in d_out -> n_tokens d_out",
        )
        return output.write_block(Y_block)

    output = tjax.fori_loop(0, n_experts.size, body, init_val=L)
    output.done()

    # Numerical check: the spec is `sum_{d_in} X[i,d_in] * W[eid[i],d_in,d_out]`.
    expected_arr = jnp.einsum(
        "nd,nde->ne", X_sorted.arr, W.arr[eid_values],
    )
    assert jnp.allclose(output.arr, expected_arr, atol=1e-5)


def test_fused_moe_fori_loop_scalar(reset):
    """The full fused-MoE loop, scalar accumulation. fori_loop over
    experts, each iter slices `X_sorted` to the expert's block via
    `block_at`, matmuls with `W[g]` (loaded once per expert), and
    contributes its block's scalar to the running total. The
    invariant is the parametric sum of per-block contributions for
    experts `[0, k)`. After the loop, the result is the per-expert
    blocked sum — same canonical form as a spec written via
    `ParametricReduce` over the same body. The block-constant
    rewrite stitches the inductive step together: the per-iter body
    uses `W[g]`, the spec form uses `gather(W, eid_sorted)`, and
    within each block they collapse to the same expression."""
    n_tokens = dim("n_tokens", 16)
    d_in = dim("d_in", 4)
    d_out = dim("d_out", 8)
    n_experts = dim("n_experts", 4)
    n_offsets = dim("n_offsets", 5)

    k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
    X_sorted = tjax.random.normal(k1, n_tokens, d_in, name="X_sorted")
    W = tjax.random.normal(k2, n_experts, d_in, d_out, name="W")

    offsets = tjax.runtime_index("offsets", n_offsets, values_in=n_tokens)
    eid_sorted = tjax.runtime_index(
        "eid_sorted", n_tokens, values_in=n_experts,
        block_sorted_paired_with="offsets",
    )

    def body(g, total):
        X_block = X_sorted.block_at(n_tokens, offsets, g)
        W_g = W.slice(n_experts, g, to_affine(g) + 1).sum(n_experts)
        Y_block = tjax.einsum(
            X_block, W_g,
            "n_tokens d_in, d_in d_out -> n_tokens d_out",
        )
        return total + Y_block.sum(n_tokens).sum(d_out)

    # Invariant: at step k, total = Σ_{g' ∈ [0, k)} per-block-scalar(g').
    # Build the per-block contribution at the inner var `g_inner`, then
    # wrap in a ParametricReduce over g_inner ∈ [0, k).
    k_sym = SymbolicInt("k")
    g_inner = SymbolicInt("g_inner")
    X_block_inner = X_sorted.block_at(n_tokens, offsets, g_inner)
    W_g_inner = W.slice(
        n_experts, g_inner, to_affine(g_inner) + 1,
    ).sum(n_experts)
    Y_block_inner = tjax.einsum(
        X_block_inner, W_g_inner,
        "n_tokens d_in, d_in d_out -> n_tokens d_out",
    )
    per_block_scalar = Y_block_inner.sum(n_tokens).sum(d_out)

    inv_type = Type(
        st=(),
        et=ParametricReduce(
            loop_var=g_inner,
            lo=0,
            hi=to_affine(k_sym),
            op="sum",
            body=per_block_scalar.type.et,
        ),
        dt=None,
    )

    total = tjax.fori_loop(
        0, n_experts.size, body,
        init_val=0.0,
        invariant=inv_type,
    )

    # Expected form: same ParametricReduce, evaluated at k = n_experts.
    # Both should be the full-loop per-expert blocked sum.
    expected_et = ParametricReduce(
        loop_var=g_inner,
        lo=0,
        hi=n_experts.size,
        op="sum",
        body=per_block_scalar.type.et,
    )
    assert verify_exprs_equivalent(total.type.et, expected_et)
