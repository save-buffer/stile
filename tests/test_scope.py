"""
Tests for `stile.scope()` — the context manager that snapshots and
restores process-wide stile state.
"""
import pytest

import stile
from stile import dim, scope, reset_stile, declare_index_properties, runtime_scalar
from stile.numerical import (
    active_hardware, WORST_CASE, NVIDIA_TENSOR_CORE_TF32, TPU_MXU,
)


def test_additive_scope_drops_new_dims(reset):
    """
    Additive `scope()` (the default) inherits the parent's registry
    and drops only what was newly registered inside the `with` block.
    """
    outer = dim("OuterDim", 32)
    assert "OuterDim" in stile.g_dim_registry

    with scope():
        inner = dim("InnerDim", 16)
        assert "InnerDim" in stile.g_dim_registry
        assert "OuterDim" in stile.g_dim_registry

    # After the with block: inner is dropped, outer persists.
    assert "InnerDim" not in stile.g_dim_registry
    assert "OuterDim" in stile.g_dim_registry


def test_clear_scope_hides_parent_state(reset):
    """
    `scope(clear=True)` starts from an empty registry. The inner
    block sees no leakage from prior tests / module-level setup.
    On exit, the prior state is fully restored.
    """
    outer = dim("ParentScopeDim", 32)
    assert "ParentScopeDim" in stile.g_dim_registry

    with scope(clear=True):
        # Parent's registry is invisible inside.
        assert "ParentScopeDim" not in stile.g_dim_registry
        inner = dim("ChildScopeDim", 16)
        assert "ChildScopeDim" in stile.g_dim_registry

    # After: parent restored, child dropped.
    assert "ParentScopeDim" in stile.g_dim_registry
    assert "ChildScopeDim" not in stile.g_dim_registry


def test_scope_restores_on_exception(reset):
    """
    `scope()` is finally-safe — a raised exception inside the block
    still restores the snapshot.
    """
    outer = dim("ExcScopeDim", 8)
    snapshot_size = len(stile.g_dim_registry)

    with pytest.raises(RuntimeError, match="boom"):
        with scope():
            dim("LeakedDim", 4)
            assert "LeakedDim" in stile.g_dim_registry
            raise RuntimeError("boom")

    assert "LeakedDim" not in stile.g_dim_registry
    assert len(stile.g_dim_registry) == snapshot_size


def test_scope_restores_runtime_scalars_and_index_props(reset):
    """
    All process-wide registries snap/restore — not just dims. Tests
    `runtime_scalar` metadata and `declare_index_properties` since
    those are the other ones a typical spec depends on.
    """
    runtime_scalar("OuterScalar", max_value=8)
    declare_index_properties("OuterPerm", "permutation")

    with scope():
        runtime_scalar("InnerScalar", max_value=4)
        declare_index_properties("InnerPerm", "partition")
        assert "InnerScalar" in stile.runtime_scalar_names()
        assert stile.index_has_property("InnerPerm", "partition")

    assert "InnerScalar" not in stile.runtime_scalar_names()
    assert not stile.index_has_property("InnerPerm", "partition")
    # Outer ones survive.
    assert "OuterScalar" in stile.runtime_scalar_names()
    assert stile.index_has_property("OuterPerm", "permutation")


def test_verified_as_context_manager_with_hardware(reset):
    """
    `with stile.verified(hardware=…):` activates BOTH a fresh scope
    AND a numerical_context. Inside, `active_hardware()` returns the
    passed model; outside, it falls back to WORST_CASE.
    """
    assert active_hardware() is WORST_CASE
    with stile.verified(hardware=NVIDIA_TENSOR_CORE_TF32):
        assert active_hardware() is NVIDIA_TENSOR_CORE_TF32
        dim("VHDim", 8)
        assert "VHDim" in stile.g_dim_registry
    # Dim cleaned up, hardware restored.
    assert "VHDim" not in stile.g_dim_registry
    assert active_hardware() is WORST_CASE


def test_verified_bare_decorator(reset):
    """`@stile.verified` (no parens) works as a decorator."""
    @stile.verified
    def body():
        dim("VBareDim", 4)
        return active_hardware()
    hw = body()
    assert hw is WORST_CASE
    assert "VBareDim" not in stile.g_dim_registry


def test_verified_parens_decorator_with_hardware(reset):
    """`@stile.verified(hardware=…)` works as a parameterized decorator."""
    @stile.verified(hardware=TPU_MXU)
    def body():
        return active_hardware()
    assert body() is TPU_MXU


def test_verified_passes_args_through_decorator(reset):
    """Decorator wrapper preserves args/kwargs/return."""
    @stile.verified
    def f(a, b, *, c):
        return (a, b, c)
    assert f(1, 2, c=3) == (1, 2, 3)


def test_verified_restores_on_exception(reset):
    """
    An exception inside the `with verified(...)` block still
    restores the numerical context AND the scope.
    """
    with pytest.raises(RuntimeError, match="boom"):
        with stile.verified(hardware=NVIDIA_TENSOR_CORE_TF32):
            dim("VExcDim", 4)
            raise RuntimeError("boom")
    assert active_hardware() is WORST_CASE
    assert "VExcDim" not in stile.g_dim_registry


def test_nested_scopes(reset):
    """
    Scopes nest. Each level snapshots and restores; nothing leaks
    out of an inner scope into its parent.
    """
    dim("L0", 32)
    with scope():
        dim("L1", 16)
        with scope():
            dim("L2", 8)
            assert {"L0", "L1", "L2"} <= set(stile.g_dim_registry.keys())
        # L2 gone, L1 still in scope.
        assert "L2" not in stile.g_dim_registry
        assert "L1" in stile.g_dim_registry
    # L1 also gone.
    assert "L1" not in stile.g_dim_registry
    assert "L0" in stile.g_dim_registry
