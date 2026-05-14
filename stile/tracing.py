"""
Framework-agnostic symbolic verification machinery for traced backends.

Eager backends (numpy, torch) build their result tensors via direct
assignment — they don't need symbolic-bound tracking. Traced backends
(jax via `fori_loop`, a future typed Triton, etc.) accumulate work
under symbolic loop variables and `tensor_element` lookups, then
discharge per-tile correctness and coverage at the end.

This module hosts the pieces those traced backends share:

- `_g_runtime_arrs` — registry of runtime tensor name → backend-native
  array. Lets `tensor_element(offsets, g)` resolve to a Python int when
  both `offsets`'s array and `g` are concrete.
- `_resolve_te_to_int` / `_bound_as_int` — slice-bound concretization.
- `_symbolic_equal`, `_substitute_bindings`, `_unroll_interval_symbolic`,
  `_unroll_interval` — symbolic interval bookkeeping used by `done()`.
- `_verify_assign_against_spec` — the slice-aware ET equivalence check
  that fires the block-constant rewrite for per-tile verification.
- `CoverageTracker` — base class for traced-backend `TypedResult`s.
  Holds the `_assigned` ledger and `done()` coverage check; subclasses
  override `_init_array(shape)` and `_write_to_arr(slices, tile_arr)`
  to bridge to their framework's array type and write idiom.
"""

import stile.type as t
from .type import (
    Type, Sliced, Reduce, dim_size, dim_start, dim_end, dim_name,
    dim_full_dim, as_int, AffineExpr, SymbolicIndex,
)
from .indexing import (
    SymbolicInt, LoopVariable, to_affine, free_vars,
    resolve_symbolic_index, LoopScope, _active_loop_scopes,
)
from .specification import parse_spec_into_type
from .verification import verify_dims_equivalent, verify_exprs_equivalent


# Registry of runtime tensor name → backend-native array. Populated by
# each backend's `runtime_index` (or equivalent). The values are just
# whatever the backend's tensor type is — we only ever do `arr[pos_int]`
# and `int(...)` on the result, which works for jnp, np, torch, etc.
_g_runtime_arrs : dict = {}


def _resolve_te_to_int(x : SymbolicIndex) -> "int | None":
    """
    Resolve a `tensor_element(name, position)` atom to a concrete
    Python int when (a) `position` is concrete, and (b) the named
    tensor has a registered runtime `arr`. Accepts either a bare
    `SymbolicInt` atom or an `AffineExpr` of the form `0 + 1 *
    te(...)` (which is what `dim_start`/`dim_end` produce when they
    add the slice offset to a base of `0`). Returns `None` otherwise —
    callers fall back to `as_int` for non-`tensor_element` cases.
    """
    if isinstance(x, AffineExpr):
        if x.const != 0 or len(x.terms) != 1:
            return None
        (atom, coeff), = x.terms
        if coeff != 1:
            return None
        x = atom
    if not isinstance(x, SymbolicInt) or x.source is None:
        return None
    tensor_name, position = x.source
    pos_int = as_int(position)
    if pos_int is None:
        return None
    arr = _g_runtime_arrs.get(tensor_name)
    if arr is None:
        return None
    return int(arr[pos_int])


def _bound_as_int(x : SymbolicIndex) -> "int | None":
    """`as_int` plus `tensor_element` resolution. Used by `assign` to
    decide whether a slice bound can be materialized to a Python int
    for the runtime write."""
    v = as_int(x)
    if v is not None:
        return v
    return _resolve_te_to_int(x)


def _bound_runtime(x : SymbolicIndex):
    """
    Resolve a `SymbolicIndex` to one of:
      - a Python `int` (purely concrete; tjax ops take their fast path),
      - a jax-array-like value (a `jax.Array` or tracer; tjax ops dispatch
        to `jax.lax.dynamic_slice` / `jax.lax.fori_loop` / etc.),
      - `None` (purely symbolic; tjax ops fall back to building only the
        symbolic type, no runtime array).

    Used by every tjax op that branches on whether its bounds are
    concrete. The three-way split is what makes `tjax.jit` work: at
    verification time everything's `None`; at `jax.jit`-traced execution
    time everything's a tracer; at plain-Python execution time everything's
    a Python int. The op chooses its execution path from the resolver's
    answer.
    """
    # Already a Python int.
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    # Bare SymbolicInt with a runtime-bound value (tracer or int).
    if isinstance(x, SymbolicInt):
        if x.runtime_value is not None:
            return x.runtime_value
        # Try tensor-element resolution via _g_runtime_arrs (when the source
        # is `tensor_element(name, pos)` with pos concrete + arr registered).
        return _resolve_te_to_int(x)
    # AffineExpr: compose runtime values term-by-term. If every term has a
    # Python-int runtime, the result is a Python int. If any term is a
    # jax tracer, the result is a jax expression. If any term is unbound,
    # the whole thing is unbound.
    if isinstance(x, AffineExpr):
        # Try the all-Python-int fast path first.
        as_int_val = as_int(x)
        if as_int_val is not None:
            return as_int_val
        # Now check tensor-element resolution for atoms with a source.
        te_val = _resolve_te_to_int(x)
        if te_val is not None:
            return te_val
        # Mixed / tracer case: build the runtime value by summing terms.
        result = x.const
        has_tracer = False
        for atom, coeff in x.terms:
            atom_rt = _bound_runtime(atom)
            if atom_rt is None:
                return None
            if not isinstance(atom_rt, int):
                has_tracer = True
            result = result + coeff * atom_rt
        return result if (has_tracer or isinstance(result, int)) else None
    # Anything else (e.g., a raw jax tracer passed directly): pass through.
    return x


def _symbolic_equal(a : SymbolicIndex, b : SymbolicIndex) -> bool:
    """
    Structural equality on `SymbolicIndex`. Resolves `tensor_element`
    atoms against declared boundaries first so that, e.g.,
    `te("offsets", 0)` and `0` compare equal when `offsets[0]` is
    declared `0`.
    """
    return to_affine(resolve_symbolic_index(a)) == to_affine(
        resolve_symbolic_index(b)
    )


def _substitute_bindings(
    aff : AffineExpr, bindings : dict,
) -> SymbolicIndex:
    """
    Substitute `bindings` (loop var → concrete int) into an
    `AffineExpr`. Walks each term; for tensor-element atoms whose
    position references a bound loop var, the substitution recurses
    into the position field so e.g. `tensor_element("offsets", g)`
    becomes `tensor_element("offsets", 5)` when `g` is bound to 5.
    Result is a `SymbolicIndex` — may still be symbolic if there are
    unbound atoms.
    """
    result : SymbolicIndex = aff.const
    for v, c in aff.terms:
        if v in bindings:
            result = to_affine(result) + bindings[v] * c
            continue
        # If `v` has a `source` whose position references a bound var,
        # produce a new tensor_element atom with the position
        # substituted.
        if v.source is not None:
            tensor_name, position = v.source
            new_position = _substitute_bindings(to_affine(position), bindings)
            new_v = SymbolicInt(
                name=v.name,
                source=(tensor_name, to_affine(new_position)),
            )
            result = to_affine(result) + c * new_v
            continue
        result = to_affine(result) + c * v
    return result


def _unroll_interval_symbolic(
    start : SymbolicIndex,
    end : SymbolicIndex,
    scopes : tuple[LoopScope, ...],
) -> list[tuple[SymbolicIndex, SymbolicIndex]]:
    """
    Like `_unroll_interval` but keeps results symbolic. For each
    relevant loop scope, substitutes the loop var with each integer
    value in its range, producing one `(start, end)` per binding.
    `tensor_element` atoms with the loop var nested inside their
    position field get the substitution pushed down via
    `_substitute_bindings`. Unbound symbolic atoms (e.g.
    `tensor_element` lookups with no nested loop var, or `te` at a
    bound position whose value isn't declared) survive — they're
    resolved later by `_verify_coverage` via boundary declarations.
    """
    start_aff, end_aff = to_affine(start), to_affine(end)
    relevant_vars = free_vars(start_aff) | free_vars(end_aff)
    relevant_scopes = [s for s in scopes if s.var in relevant_vars]

    if not relevant_scopes:
        return [(start, end)]

    bindings_list : list[dict] = [{}]
    for scope in relevant_scopes:
        lo, hi = as_int(scope.lo), as_int(scope.hi)
        if lo is None or hi is None:
            # Symbolic loop bound — can't enumerate. Keep one symbolic
            # interval; coverage check will inspect it via boundaries.
            return [(start, end)]
        expanded = []
        for bindings in bindings_list:
            for k in range(lo, hi):
                extended = dict(bindings)
                extended[scope.var] = k
                expanded.append(extended)
        bindings_list = expanded

    out : list[tuple[SymbolicIndex, SymbolicIndex]] = []
    for b in bindings_list:
        out.append((
            _substitute_bindings(start_aff, b),
            _substitute_bindings(end_aff, b),
        ))
    return out


def _unroll_interval(
    start : SymbolicIndex,
    end : SymbolicIndex,
    scopes : tuple[LoopScope, ...],
) -> list[tuple[int, int]]:
    """
    Produce the list of concrete `(start, end)` integer intervals that
    `(start, end)` takes on as the recorded loop scopes range over their
    iteration domains. Only scopes whose loop variable actually appears in
    the bounds are unrolled, and their ranges must be concrete integers.
    """
    start_aff, end_aff = to_affine(start), to_affine(end)
    relevant_vars = free_vars(start_aff) | free_vars(end_aff)
    relevant_scopes = [s for s in scopes if s.var in relevant_vars]

    if not relevant_scopes:
        s_i, e_i = as_int(start), as_int(end)
        if s_i is None or e_i is None:
            raise ValueError(
                f"Cannot verify coverage for bounds ({start}, {end}): "
                "they involve loop variables not recorded at assign-time."
            )
        return [(s_i, e_i)]

    bindings_list : list[dict] = [{}]
    for scope in relevant_scopes:
        lo, hi = as_int(scope.lo), as_int(scope.hi)
        if lo is None or hi is None:
            raise ValueError(
                f"Cannot verify coverage: loop {scope.var.name!r} has "
                f"symbolic bounds ({scope.lo}, {scope.hi})."
            )
        expanded = []
        for bindings in bindings_list:
            for k in range(lo, hi):
                extended = dict(bindings)
                extended[scope.var] = k
                expanded.append(extended)
        bindings_list = expanded

    out : list[tuple[int, int]] = []
    for b in bindings_list:
        s_val = as_int(_substitute_bindings(start_aff, b))
        e_val = as_int(_substitute_bindings(end_aff, b))
        out.append((s_val, e_val))
    return out


def _verify_assign_against_spec(spec_type : Type, tile_type : Type) -> bool:
    """
    Verify a `TypedResult.assign` call: the assigned tile's type must
    match the spec, but where the tile carries `Sliced` dims (i.e. it's
    a per-block slice of the eventual output), the comparison happens
    *within the slice*. The block-constant rewrite and any other
    slice-aware normalizations only fire under a `Reduce` carrying the
    slice in its `dim` field, so we wrap both sides in an auxiliary
    `Reduce(Sliced(...), sum, ...)` for each sliced dim of the tile.
    The sum is a carrier, not a real reduction — both sides get the
    same wrap, so if their bodies are equivalent under the slice
    context, the wrapped expressions are too.

    For dims of the tile that are *not* sliced (i.e. shared with the
    spec at full extent), no wrap is needed; raw ET equality suffices.
    """
    if not verify_dims_equivalent(spec_type.st, tile_type.st):
        return False
    spec_et = spec_type.et
    tile_et = tile_type.et
    for d in tile_type.st:
        if isinstance(d, Sliced):
            spec_et = t.Reduce(op="sum", dim=d, child=spec_et)
            tile_et = t.Reduce(op="sum", dim=d, child=tile_et)
    return verify_exprs_equivalent(spec_et, tile_et)


class CoverageTracker:
    """
    Base class for traced-backend `TypedResult`s. Holds the spec, the
    `_assigned` ledger (per-tile bounds + active loop scopes at
    assign-time), and the `done()` coverage check.

    Subclasses bridge to a framework's array type by overriding two
    hooks:
      - `_init_array(shape)`: return a zero-initialized array of the
        given shape.
      - `_write_to_arr(slices, tile_arr)`: write `tile_arr` into
        `self.arr` at the given Python slice tuple (mutating `self.arr`
        or rebinding it, depending on the framework's idiom).

    The eager numpy/torch backends keep their own ad-hoc `TypedResult`s
    — they don't need any of this machinery.
    """
    def __init__(self, spec : str):
        self.expected_type = parse_spec_into_type(spec)
        shape_dims = self.expected_type.st if self.expected_type.st is not None else ()
        self.shape = tuple(as_int(dim_size(d)) or 0 for d in shape_dims)
        self.arr = self._init_array(self.shape)
        # Each entry is (per-dim bounds tuple, snapshot of active LoopScopes
        # at assign-time). `done()` uses the scope snapshot to unroll
        # symbolic bounds into concrete intervals for coverage checking.
        self._assigned : list[tuple[
            tuple[tuple[SymbolicIndex, SymbolicIndex], ...],
            tuple[LoopScope, ...],
        ]] = []

    def _init_array(self, shape):
        raise NotImplementedError

    def _write_to_arr(self, slices, tile_arr):
        raise NotImplementedError

    def assign(self, result):
        if not _verify_assign_against_spec(self.expected_type, result.type):
            raise ValueError(
                "Attempted to assign a tensor that does not match the spec! "
                f"Expected: {self.expected_type.et}, actual: {result.type.et}"
            )

        per_dim_bounds = tuple(
            (dim_start(d), dim_end(d)) for d in result.type.st
        )
        scopes_snapshot = tuple(_active_loop_scopes)
        self._assigned.append((per_dim_bounds, scopes_snapshot))

        # Materialize the runtime write when every bound resolves to a
        # Python int. Bounds that came from `tensor_element(offsets, g)`
        # resolve via `_g_runtime_arrs` when both `g` and `offsets`'s
        # arr are concrete — that's the path the carry pattern relies on.
        concrete_bounds = tuple(
            (_bound_as_int(s), _bound_as_int(e)) for s, e in per_dim_bounds
        )
        all_concrete_bounds = all(
            s is not None and e is not None for s, e in concrete_bounds
        )
        if result.arr is not None and all_concrete_bounds:
            slices = tuple(slice(s, e) for s, e in concrete_bounds)
            self._write_to_arr(slices, result.arr)

    def write_block(self, tile) -> "CoverageTracker":
        """
        Carry-pattern analog of `assign`: verifies the tile against the
        spec restricted to the tile's slice, records its bounds for
        coverage, and returns `self` so the tracker can be threaded as
        the `fori_loop` carry. The tile's own type already encodes the
        block (via `Sliced` dims from `block_at`), so no explicit (dim,
        offsets, g) args are needed.
        """
        self.assign(tile)
        return self

    def done(self):
        """
        Verify that the recorded `assign` calls exactly tile the full
        output shape along every dimension — no gaps, no overlaps.

        Concrete bounds (from assigns in a static-bound fori_loop) are
        unrolled into integer intervals and a sorted-cursor sweep
        verifies coverage. Symbolic bounds (from assigns whose slice
        bounds are `tensor_element`-derived, e.g. `offsets[g]` inside a
        per-expert fori_loop) stay symbolic; their adjacency is checked
        structurally between successive unrolled iterations, and their
        first-start / last-end are checked against `tensor_boundary`
        declarations to confirm coverage of the full dim.
        """
        for dim_idx, d in enumerate(self.expected_type.st):
            full_size = as_int(dim_size(d))
            if full_size is None:
                continue
            unrolled : list[tuple[SymbolicIndex, SymbolicIndex]] = []
            for bounds, scopes in self._assigned:
                s, e = bounds[dim_idx]
                unrolled.extend(_unroll_interval_symbolic(s, e, scopes))
            if not unrolled:
                continue
            self._verify_coverage(d, unrolled, full_size)

    def _verify_coverage(
        self,
        d,
        intervals : "list[tuple[SymbolicIndex, SymbolicIndex]]",
        full_size : int,
    ) -> None:
        """Sort-and-sweep coverage check that accepts symbolic
        `(start, end)` pairs. Pure-int intervals get the original
        cursor sweep; mixed sets are normalized by resolving
        `tensor_element` bounds against declared boundaries, then
        either concretized fully or checked symbolically: adjacent
        intervals must satisfy `prev.end == next.start`
        structurally, and the chain must start at `0` and end at
        `full_size`."""
        # Dedupe: repeat assigns with identical bounds (common for dims
        # that aren't tiled — every tile covers the full extent) are fine.
        intervals = list({(s, e): None for s, e in intervals}.keys())
        resolved = [
            (resolve_symbolic_index(s), resolve_symbolic_index(e))
            for s, e in intervals
        ]
        def sort_key(iv):
            s_int = as_int(iv[0])
            return (0, s_int) if s_int is not None else (1, repr(iv[0]))
        resolved.sort(key=sort_key)
        cursor : SymbolicIndex = 0
        for s, e in resolved:
            if not _symbolic_equal(s, cursor):
                raise ValueError(
                    f"Dimension {dim_name(d)!r} has a gap or overlap: "
                    f"cursor at {cursor}, next interval starts at {s}"
                )
            cursor = e
        if not _symbolic_equal(cursor, full_size):
            raise ValueError(
                f"Dimension {dim_name(d)!r} only covered up to {cursor} "
                f"of {full_size}"
            )
