"""
Shared pytest fixtures for the stile test suite.

The `reset` fixture wraps each test in a `stile.scope()` block so all
process-wide stile state (declared dims, runtime scalars, index
properties, …) declared during the test is snapshot-restored on exit
— including on exceptions. Replaces the older `yield; reset_stile()`
pattern, which had two issues:

  1. If the test raised before the fixture's post-yield code ran,
     state leaked into subsequent tests.
  2. Module-level dim declarations (`M = dim("M", 32)` at top of a
     test file) got wiped between tests, so any test that depended
     on them had to manually re-register at the top of its body.

`scope()` is additive by default — module-level setup persists across
tests; only what the test itself registers is rolled back. Tests that
want hermetic isolation can use `scope(clear=True)` explicitly inside
their body.
"""
import pytest

import stile


@pytest.fixture
def reset():
    with stile.scope():
        yield


# --- Shared skipif decorators ------------------------------------------
# Centralized so per-file copies don't drift; importable from any test
# module as `from conftest import REQUIRES_TORCH, REQUIRES_JAX,
# REQUIRES_CUDA, REQUIRES_TRITON`.
try:
    import torch as _torch
    HAS_TORCH = True
    HAS_CUDA = _torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

try:
    import jax as _jax  # noqa: F401
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import stile.triton as _stile_triton
    HAS_TRITON = _stile_triton.HAS_TRITON
except ImportError:
    HAS_TRITON = False


REQUIRES_TORCH = pytest.mark.skipif(
    not HAS_TORCH, reason="torch not installed",
)
REQUIRES_JAX = pytest.mark.skipif(
    not HAS_JAX, reason="jax not installed",
)
REQUIRES_CUDA = pytest.mark.skipif(
    not (HAS_TORCH and HAS_CUDA), reason="needs torch + CUDA (run on spark)",
)
REQUIRES_TRITON = pytest.mark.skipif(
    not HAS_TRITON, reason="Triton not installed",
)
