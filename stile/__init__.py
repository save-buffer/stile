# Public API. Third-party DSLs `import stile` and build wrappers over
# these primitives — the type algebra, the spec parser, the verifier,
# and the indexing/domain helpers. Internal helpers (the `_g_*`
# registries, the normalizer, etc.) are accessible via the submodules
# but aren't surfaced here.

# --- Types & ET nodes ---------------------------------------------------
from .type import (
    Type, ShapeType, FullDim, Sliced, Tensor, Constant, TagCond,
    BinaryOp, UnaryOp, Repeat, Reduce, Gather, Scatter,
    BinaryOpType, ReduceOpType, ExprType, Dim, SymbolicIndex,
    g_dim_registry, _reset_tensor_counter,
    # ET-builder helpers — same lowerings the spec parser uses.
    exp, sin, cos, sqrt, maximum, minimum, abs, relu,
    einsum, type_from_binary_op, override_dims_in_type,
    substitute_tensor_in_et,
    dim_name, dim_size, dim_full_dim, dim_contains, dim_start, dim_end,
    simplify_dim, as_int,
)

# --- Spec parser --------------------------------------------------------
from .specification import (
    parse_spec_into_type, parse_predicate, LexState,
)

# --- Verifier (normalizer, equality, substitution) ----------------------
from .verification import (
    verify_exprs_equivalent, verify_types_equivalent, verify_dims_equivalent,
    diff_exprs,
    normalize, substitute_lv_in_expr, substitute_loop_var_in_et,
    simplify_under_active_loop_scope,
    NormalizedExpr, NormalizedProduct, NormalizedTensor, NormalizedTagCond,
    NormalizedRepeat, NormalizedReduce, NormalizedSum, NormalizedMax,
    NormalizedGather, NormalizedScatter, NormalizedExp, NormalizedUnaryOp,
    NormalizedFactor,
    make_tag_cond, make_max, make_sum, make_expr, make_reduce,
    varies_with_dim,
)

# --- Indexing & domains -------------------------------------------------
from .indexing import (
    SymbolicInt, LoopVariable, AffineExpr, AffineConstraint,
    Domain, range_domain, domain, and_domains, or_domains,
    le, lt, ge, gt, eq,
    LoopScope, loop, active_loop_domain, active_loop_vars,
    RuntimeScalar, runtime_scalar, runtime_scalar_max, runtime_scalar_names,
    SymInfo, symint_info,
    tensor_element, to_affine, free_vars, evaluate,
    declare_index_properties, index_has_property,
    declare_block_pairing, paired_index_for_offsets,
    declare_tensor_boundary, tensor_boundary, resolve_symbolic_index,
    # Module-level registries are still importable from `stile.indexing`
    # for stile-internal use (e.g. `reset_stile`), but DSLs should use
    # the typed accessors above instead.
    _active_loop_scopes,
    _g_runtime_scalars, _g_symint_metadata, _g_index_properties,
    _g_block_pairings, _g_tensor_boundaries,
)
from .tracing import _g_runtime_arrs

# --- Numerical analysis -------------------------------------------------
# Reexport the active-hardware accessor + context so `stile.verified`
# can wrap a numerical context without inline imports. The full
# `stile.numerical.*` surface (AffineForm, evaluator, presets) stays
# in the submodule.
from .numerical import (
    WORST_CASE as _DEFAULT_NUMERICS,
    numerical_context as _numerical_context,
)

def dim(name : str, size : int) -> FullDim:
    return FullDim(name, size)

def expr_simplifies(
    expr : Type,
    spec : str,
) -> bool:
    spec_type = parse_spec_into_type(spec)
    return verify_exprs_equivalent(expr.type.et, spec_type.et)


# --- Global state management ------------------------------------------
# Stile keeps a handful of process-wide registries (declared dims,
# runtime scalars, active loop scopes, index properties, …) so that
# spec strings can reference them by name. The functions below let
# callers snapshot and restore that state cleanly.

# Each entry is `(container, snapshot_fn, restore_fn)` where `container`
# is one of the actual module-level dicts/lists. `snapshot_fn` returns
# a deep-enough copy; `restore_fn` repopulates the container in place
# (re-binding the module attribute wouldn't work — other modules hold
# references to the originals).
def _scope_registries():
    """The set of process-wide registries that participate in
    `scope()` snapshot/restore. Returned as a tuple of containers so
    helpers can iterate (each is a dict or list — mutated in-place;
    rebinding the module attribute wouldn't reach existing imports)."""
    return (
        g_dim_registry,
        _active_loop_scopes,
        _g_runtime_scalars,
        _g_symint_metadata,
        _g_index_properties,
        _g_block_pairings,
        _g_tensor_boundaries,
        _g_runtime_arrs,
    )


def _snapshot_state() -> dict:
    from .type import _g_tensor_counter
    return {
        "registries": [
            dict(r) if isinstance(r, dict) else list(r)
            for r in _scope_registries()
        ],
        "tensor_counter": _g_tensor_counter[0],
    }


def _restore_state(snapshot : dict) -> None:
    from .type import _g_tensor_counter
    for container, snapped in zip(_scope_registries(), snapshot["registries"]):
        container.clear()
        if isinstance(container, dict):
            container.update(snapped)
        else:
            container.extend(snapped)
    _g_tensor_counter[0] = snapshot["tensor_counter"]


def _clear_state() -> None:
    """Wipe every registry. Used internally by `reset_stile()` and
    `scope(clear=True)`."""
    for container in _scope_registries():
        container.clear()
    _reset_tensor_counter()


def reset_stile() -> None:
    """Wipe all process-wide stile state. Equivalent to entering a
    `scope(clear=True)` block and never exiting; prefer `scope(...)`
    for tests / library code that wants automatic restoration."""
    _clear_state()


import contextlib
import functools


class _Verified:
    """
    The object behind `stile.verified` — works as both a context
    manager (`with stile.verified(...):`) and a decorator
    (`@stile.verified`). Bundles `stile.scope()` (so dim / runtime-
    scalar / index-property registrations are rolled back on exit)
    with `stile.numerical.numerical_context(hardware=...)` (so AA
    ops use the matching hardware-numerics model).

    Don't construct directly; use the `verified` factory below.
    """
    def __init__(self, *, hardware=None, clear : bool = False):
        self._hardware = hardware if hardware is not None else _DEFAULT_NUMERICS
        self._clear = clear
        self._scope_ctx = None
        self._numerics_ctx = None

    def __enter__(self):
        self._scope_ctx = scope(clear=self._clear)
        self._numerics_ctx = _numerical_context(hardware=self._hardware)
        self._scope_ctx.__enter__()
        self._numerics_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._numerics_ctx.__exit__(exc_type, exc, tb)
        self._scope_ctx.__exit__(exc_type, exc, tb)
        return False

    def __call__(self, fn):
        hardware = self._hardware
        clear = self._clear
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _Verified(hardware=hardware, clear=clear):
                return fn(*args, **kwargs)
        return wrapper


def verified(fn=None, *, hardware=None, clear : bool = False):
    """
    Unified verification entry — works as decorator, parameterized
    decorator, or context manager. Bundles `stile.scope()` (dim /
    runtime-scalar / index-property cleanup) with
    `numerical_context(hardware=...)` (AA ops use the matching
    hardware-numerics model). Defaults to `WORST_CASE` numerics
    when no `hardware=` is given.

    Four call shapes, all equivalent up to defaults:

        @stile.verified                              # bare decorator
        @stile.verified()                            # parens, no args
        @stile.verified(hardware=NVIDIA_TF32_MATMUL) # with hardware
        with stile.verified(hardware=...): ...       # context manager

    Inside the block:
      - dims declared via `stile.dim(...)` are dropped on exit;
      - runtime scalars, index properties, etc. are restored;
      - AA on typed values uses `hardware`'s numerics (matmul
        accumulator dtype, reduction order, …).
    """
    if fn is None:
        return _Verified(hardware=hardware, clear=clear)
    if callable(fn):
        return _Verified(hardware=hardware, clear=clear)(fn)
    raise TypeError(
        f"verified: positional arg must be a callable to decorate; "
        f"got {type(fn).__name__}. Did you mean to pass `hardware=` "
        f"as a keyword? `verified(hardware=...)`."
    )


@contextlib.contextmanager
def scope(*, clear : bool = False):
    """
    Scoped stile state. Snapshots every process-wide registry on entry
    and restores it on exit (even on exception). Two modes:

    - **Additive** (default): inherit the parent's dim registry,
      runtime scalars, index properties, etc. Anything declared inside
      the `with` block is dropped on exit, but pre-existing entries
      pass through.
    - **Fresh** (`clear=True`): start from an empty registry. Useful
      for hermetic test isolation — the inner block sees no leakage
      from prior tests / module-level setup.

    Typical pytest pattern:

        @pytest.fixture
        def reset():
            with stile.scope():
                yield

    Library code that wants to verify a one-off expression without
    polluting the caller's registry:

        with stile.scope():
            M = stile.dim("M", 32)
            ...verify here...
        # After: M is no longer registered.
    """
    snapshot = _snapshot_state()
    if clear:
        _clear_state()
    try:
        yield
    finally:
        _restore_state(snapshot)


def mask_expr(
    dims : tuple[FullDim, ...],
    domain : Domain,
) -> Tensor:
    """
    A tagged Tensor whose value is `1` on positions in `domain` and `0`
    elsewhere. The tag is `Cond(domain, Value(1), Value(0))`. `domain`'s
    constraints should reference `LoopVariable`s named after the tensor's
    dims — those are the symbolic dim-indices.

    Common masks (causal, band, block-diagonal) are library wrappers over
    this primitive, produced by constructing the appropriate `Domain`.
    """
    tag = TagCond(
        domain=domain,
        if_true=Constant(1.0),
        if_false=Constant(0.0),
    )
    # Mask tensors use a fixed name so two `mask_expr` calls with the
    # same dims+predicate produce equal tensors (the tag carries the
    # identifying info).
    return Tensor(dims=dims, tag=tag, name="_mask")


