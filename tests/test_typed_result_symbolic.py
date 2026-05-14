"""
`TypedResult.done()` working through fori_loops whose slice bounds
are runtime `tensor_element` lookups (e.g. `offsets[g]`). The
verifier resolves the boundary `tensor_element`s (declared via
`boundary_values=`) and checks symbolic adjacency between successive
iterations to confirm the per-block writes tile the full output.

This is the coverage half of the fused-MoE story: the verifier
already knows kernel-vs-spec equivalence inside each block (the
block-constant rewrite); `done()` now verifies that the per-block
assigns line up end-to-end across the full output dim, with the
loop's first iteration starting at `0` and its last ending at
`dim.size`.
"""
import jax
import pytest

import stile.jax as tjax
from stile import dim, reset_stile


@pytest.fixture
def reset():
    yield
    reset_stile()


def test_done_through_symbolic_block_loop(reset):
    """fori_loop over experts, per iter assigns a slice of X
    bounded by `(offsets[g], offsets[g+1])` to the output buffer.
    `done()` unrolls g over [0, n_experts), keeps the
    `tensor_element` bounds symbolic, sweeps them in order, and
    resolves the boundary lookups (`offsets[0]==0`,
    `offsets[n_experts]==n_tokens.size`) to confirm full coverage."""
    n_tokens = dim("n_tokens", 16)
    d = dim("d", 4)
    n_offsets = dim("n_offsets", 5)  # n_experts + 1
    n_experts = dim("n_experts", 4)

    X = tjax.random.normal(jax.random.PRNGKey(0), n_tokens, d, name="X")
    offsets = tjax.runtime_index(
        "offsets", n_offsets, values_in=n_tokens,
        boundary_values={0: 0, n_experts.size: n_tokens.size},
    )

    L = tjax.TypedResult("X:n_tokens d")

    def body(g, _):
        X_block = X.block_at(n_tokens, offsets, g)
        L.assign(X_block)
        return None

    tjax.fori_loop(0, n_experts.size, body, None)
    L.done()


def test_done_rejects_block_loop_with_gap(reset):
    """If the loop covers only the first three blocks (the last
    block goes unassigned), `done()` should refuse — the last
    interval's end is `offsets[3]`, which the boundary declarations
    don't resolve to `n_tokens.size`."""
    n_tokens = dim("n_tokens", 16)
    d = dim("d", 4)
    n_offsets = dim("n_offsets", 5)
    n_experts = dim("n_experts", 4)

    X = tjax.random.normal(jax.random.PRNGKey(0), n_tokens, d, name="X")
    offsets = tjax.runtime_index(
        "offsets", n_offsets, values_in=n_tokens,
        boundary_values={0: 0, n_experts.size: n_tokens.size},
    )

    L = tjax.TypedResult("X:n_tokens d")

    def body(g, _):
        X_block = X.block_at(n_tokens, offsets, g)
        L.assign(X_block)
        return None

    # Only iterate 3 of 4 blocks — the 4th block's slice never gets
    # assigned, so the last covered end is `offsets[3]` rather than
    # `offsets[4] == n_tokens.size`.
    tjax.fori_loop(0, 3, body, None)
    with pytest.raises(ValueError, match="only covered up to"):
        L.done()
