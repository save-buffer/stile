"""
Runtime-indexed `gather` — opaque to the verifier. The kernel can read
through a runtime index tensor (paged KV cache, token routing,
etc.); the verifier matches structurally as long as kernel and spec
reference the same index tensor.

The slice-pushdown identity is the load-bearing one: a kernel that
slices the *gathered* tensor per-iteration produces the same
normalized form as the spec that slices the *index* once. Without
that, a tile-walked paged-attention kernel can't fuse against its
gather-form spec.
"""
import jax
import jax.numpy as jnp

import stile.jax as tjax
from stile import dim, reset_stile
from stile.verification import verify_exprs_equivalent
import pytest


@pytest.fixture
def reset():
    yield
    reset_stile()


def test_gather_same_idx_structural_equality(reset):
    """
    Two `gather`s with the same source, dim, and idx tensor produce
    structurally identical expressions — the verifier doesn't need to
    inspect `idx`'s values.
    """
    N_phys = dim("N_phys", 32)
    D = dim("D", 4)
    N_log = dim("N_log", 16)
    K_pool = tjax.random.normal(jax.random.PRNGKey(0), N_phys, D, name="K_pool")
    page_table = tjax.runtime_index("page_table", N_log, values_in=N_phys)

    a = K_pool.gather(N_phys, page_table)
    b = K_pool.gather(N_phys, page_table)
    assert verify_exprs_equivalent(a.type.et, b.type.et)


def test_gather_tile_sum_invariant(reset):
    """
    The load-bearing test: tile-walk a 2-d gathered tensor and verify
    the running sum matches the full gather-sum. Wired through
    `@tjax.jit` so the same code is type-checked once and then
    executed under `jax.jit` for a numerical check. Exercises:
      - slice of gather ≡ gather of sliced idx (the slice info ends up
        in the surrounding Reduce's `Sliced` dim, identical on both
        sides),
      - tile-merge over symbolic intervals against an opaque-leaf
        gather body (existing tile-merge in `_merge_sum_reduces` keys
        off `(dim, child, extras)`; with gather as the body, the
        child is just the `NormalizedGather` factor — same on both
        tiles and on the spec),
      - sum-pull through the outer `D` reduce so `sum_D(sum_N_tile_a) +
        sum_D(sum_N_tile_b)` collapses to `sum_D(sum_N_full)`.
    """
    N_phys = dim("N_phys", 32)
    D = dim("D", 4)
    N_log = dim("N_log", 8)
    BN = 4
    K_pool = tjax.random.normal(jax.random.PRNGKey(0), N_phys, D, name="K_pool")
    page_table = tjax.runtime_index("page_table", N_log, values_in=N_phys)

    @tjax.jit(
        spec="sum[D](sum[N_log]("
             "gather[N_phys](K_pool:N_phys D, page_table:N_log)"
             ")) -> "
    )
    def gather_tile_sum(K_pool, page_table):
        K_logical = K_pool.gather(N_phys, page_table)
        def body(k, acc):
            K_tile = K_logical.slice(N_log, k * BN, (k + 1) * BN)
            return acc + K_tile.sum(N_log).sum(D)
        return tjax.fori_loop(
            0, N_log.size // BN, body, init_val=0.0,
            invariant=(
                f"sum[D](sum[N_log where N_log < {BN} * k]("
                f"gather[N_phys](K_pool:N_phys D, page_table:N_log)"
                f"))"
            ),
        )

    result = gather_tile_sum(K_pool=K_pool, page_table=page_table)
    expected = jnp.sum(K_pool.arr[page_table.arr])
    assert jnp.allclose(result.arr, expected, atol=1e-4)
