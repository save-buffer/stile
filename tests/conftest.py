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
