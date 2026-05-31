"""
TypedPallas — `stile.jax.pallas` is the typed wrapper around
`jax.experimental.pallas`. Same type discipline as `stile.jax`: every
value carries a `Type` (ShapeType + ExprType), and a per-assignment
verifier proves the kernel matches a one-line spec.

Distinct from `stile.jax`'s `TypedResult` because Pallas kernels run
inside a JIT trace where inputs and outputs are `Ref`s (mutable cells)
rather than values. The discipline:

  - Inputs and outputs are wrapped as `TypedRef` / `TypedOutputRef`.
  - `.load()` reads a `Ref` and produces a `TypedJaxArray` carrying the
    same `Type`. Once you've loaded, all the `tjax` ops compose normally
    — slice, einsum, exp, sum, where, mask, fori_loop.
  - `.assign(value)` on the output ref runs the verifier against the
    `OutputSpec` and stores. Same `verify_types_equivalent` machinery
    `tjax.TypedResult.assign` uses.

For local development, `interpret=True` runs Pallas on CPU with the
same trace as the GPU/TPU path. The verifier sees identical ASTs
either way.
"""
try:
    import jax
    import jax.numpy as jnp
    import jax.experimental.pallas as pl
except ImportError:
    raise ImportError(
        "Pallas support requires the jax extra: pip install stile[jax]"
    ) from None

from dataclasses import dataclass
from typing import Callable

from ...type import (
    Type, ShapeType, DataType, Sliced, dim_size, dim_full_dim, as_int,
    simplify_dim, override_dims_in_type,
)
from ...indexing import LoopVariable
from ...specification import parse_spec_into_type
from ...verification import verify_types_equivalent
from ...reference import run_reference, check_output_against_declaration
from .._core import (
    TypedJaxArray, loop_var_binding, _g_active_tile_overrides,
    dtype_to_datatype,
)


@dataclass
class OutputSpec:
    """
    Declares the *expected* output of a typed Pallas kernel:

      - `spec`: either a stile-spec-language string describing the value
        the output should hold (e.g. `"2 * N"`), OR a **jax reference
        function** — a callable over the kernel's typed inputs
        (positional, in call order) returning the expected
        `TypedJaxArray` (e.g. `lambda x: x * 2`). A reference is run on
        the actual call-time inputs to extract its `ExprType`; no spec
        string needed.
      - `st`: the output's `ShapeType` (the dim signature, possibly
        sliced). Doesn't have to match `spec`'s shape verbatim — slice
        overrides happen via `verify_types_equivalent`.
      - `dt`: the output's dtype.
    """
    spec : "str | Callable"
    st : ShapeType
    dt : DataType | None = None


class TypedRef:
    """
    A Pallas `Ref` paired with a `Type`. `.load()` reads the entire ref
    and returns a `TypedJaxArray` carrying the same `Type` — so the
    loaded value's expression is the input's expression and downstream
    `tjax` ops compose normally.
    """
    def __init__(self, ref, type : Type):
        self.ref = ref
        self.type = type

    def load(self) -> TypedJaxArray:
        return TypedJaxArray(self.ref[...], self.type)


class TypedOutputRef(TypedRef):
    """
    The output ref of a typed Pallas kernel. `.assign(value)` runs the
    verifier against the expected `Type` and stores; if the `value`'s
    expression doesn't normalize to the expected one, raises before the
    store. The expected `Type` is resolved up front by `typed_pallas_call`
    (from a spec string or a reference function), so this just compares.
    """
    def __init__(self, ref, type : Type, output_spec : OutputSpec):
        super().__init__(ref, type)
        self.output_spec = output_spec

    def assign(self, value : TypedJaxArray):
        if not verify_types_equivalent(self.type, value.type):
            raise ValueError(
                "Pallas output does not match spec! "
                f"Expected: {self.type.et}, actual: {value.type.et}"
            )
        self.ref[...] = value.arr


def _resolve_spec_et(out_type : OutputSpec, typed_inputs):
    """Resolve an `OutputSpec` to the expected output `ExprType`.

    For a spec string, parse it. For a reference function, run it on the
    call-time inputs (re-wrapped so their stile `Type` carries the real
    array dtype, making the reference's output dtype checkable) and verify
    the reference's output ShapeType / dtype against the declaration before
    returning its ExprType."""
    if not callable(out_type.spec):
        return parse_spec_into_type(out_type.spec).et
    ref_inputs = [
        TypedJaxArray(
            ti.arr,
            Type(ti.type.st, ti.type.et, dtype_to_datatype(ti.arr.dtype)),
        )
        for ti in typed_inputs
    ]
    outs = run_reference(out_type.spec, ref_inputs)
    if len(outs) != 1:
        raise ValueError(
            "typed_pallas_call: a Pallas reference must return a single "
            f"typed value (single-output kernel); got {len(outs)}."
        )
    ref_type = outs[0].type
    check_output_against_declaration(
        ref_type, out_type.st, dtype_to_datatype(out_type.dt),
        label="typed_pallas_call reference",
    )
    return ref_type.et


def _block_sliced_dim(parent_dim, block_idx, block_size):
    """
    Combine a parent dim with a block-relative position and size to
    produce the sliced dim the in-kernel ref actually exposes:
    `[block_idx * block_size, block_idx * block_size + block_size)`.
    `block_idx` is whatever `BlockSpec.index_map(*pids)` returned for
    this axis — a plain `int` for static positions, an `AffineExpr`
    when `program_id` LoopVariables flowed through. `simplify_dim`
    folds the trivial `Sliced(D, 0, D.size) → D` case.
    """
    abs_start = block_idx * block_size
    abs_end = abs_start + block_size
    return simplify_dim(Sliced(parent_dim, abs_start, abs_end))


def _derive_sliced_st(parent_st, block_spec, pids):
    """
    Build the ShapeType the kernel-side sees for a ref governed by
    `block_spec` under `pids` (one `LoopVariable` per grid axis).
    """
    block_indices = block_spec.index_map(*pids)
    block_shape = block_spec.block_shape
    sliced = []
    for parent_dim, idx, sz in zip(parent_st, block_indices, block_shape):
        sliced.append(_block_sliced_dim(parent_dim, idx, sz))
    return tuple(sliced)


def typed_pallas_call(
    kernel_fn,
    out_type : OutputSpec,
    *,
    grid=None,
    in_specs=None,
    out_specs=None,
    interpret : bool = True,
    compiler_params=None,
):
    """
    Wrap a Pallas kernel function with stile typing. The user's
    `kernel_fn` takes `TypedRef`s for each input followed by a
    `TypedOutputRef`. The returned callable takes `TypedJaxArray`
    inputs and produces a `TypedJaxArray` output.

    For tiled kernels, pass `grid` (a tuple of grid axis sizes) and
    `in_specs` / `out_specs` (raw `pl.BlockSpec`s, one per input /
    output). Inside the kernel, each ref is already block-sliced by
    Pallas; stile derives the ref's `Type` by feeding `LoopVariable`s
    into the BlockSpec's `index_map` so the type's ShapeType reflects
    the symbolic slice. The output ref's expected type is the spec
    restricted to the tile via `override_dims_in_type` — so per-block
    `assign(...)` certifies "this block matches the spec's tile."

    `interpret=True` (default) runs the kernel on CPU via Pallas's
    interpreter. Same trace as the GPU/TPU path, so the verifier sees
    identical ASTs — let the dev loop be local, the perf loop be
    remote.
    """
    out_shape_dims = tuple(as_int(dim_size(d)) for d in out_type.st)
    out_dtype = out_type.dt if out_type.dt is not None else jnp.float32
    out_struct = jax.ShapeDtypeStruct(out_shape_dims, out_dtype)

    tiled = grid is not None

    def runner(*typed_inputs : TypedJaxArray):
        # Resolve the expected output ExprType once — from a spec string or
        # by running the reference on these inputs (which also checks the
        # reference's output ShapeType / dtype against the declaration).
        spec_et = _resolve_spec_et(out_type, typed_inputs)

        def jax_kernel(*refs):
            input_refs = refs[:len(typed_inputs)]
            output_ref = refs[len(typed_inputs)]
            if tiled:
                pids = tuple(
                    LoopVariable(f"_pid_{i}") for i in range(len(grid))
                )
                # Bind each `_pid_<i>` LoopVariable to its corresponding
                # `pl.program_id(i)` jax value so symbolic slice offsets
                # used by `tjax.mask` (or any other in-kernel runtime
                # construction) can be evaluated at trace time.
                pid_runtime = {
                    f"_pid_{i}": pl.program_id(i) for i in range(len(grid))
                }
                wrapped_inputs = []
                for ref, ti, spec in zip(input_refs, typed_inputs, in_specs):
                    sliced_st = _derive_sliced_st(ti.type.st, spec, pids)
                    wrapped_inputs.append(
                        TypedRef(ref, Type(sliced_st, ti.type.et, ti.type.dt))
                    )
                # Output: the spec describes the FULL output; restrict it
                # to the tile via override_dims_in_type so per-block
                # assign certifies the tile, not the global tensor.
                sliced_out_st = _derive_sliced_st(out_type.st, out_specs, pids)
                full_spec_type = Type(out_type.st, spec_et, out_type.dt)
                tile_spec = override_dims_in_type(full_spec_type, *sliced_out_st)
                wrapped_output = TypedOutputRef(
                    output_ref,
                    Type(sliced_out_st, tile_spec.et, out_type.dt),
                    out_type,
                )
                # The tile context: every Sliced dim that the kernel
                # body operates on. Inner `tjax.fori_loop(..., invariant=...)`
                # calls read this stack to restrict their parsed
                # invariant types to the tile, so the body's Sliced
                # types match the invariant's during verification.
                tile_overrides = tuple(
                    d for d in sliced_out_st if isinstance(d, Sliced)
                )
                _g_active_tile_overrides.append(tile_overrides)
                try:
                    with loop_var_binding(pid_runtime):
                        kernel_fn(*wrapped_inputs, wrapped_output)
                finally:
                    _g_active_tile_overrides.pop()
            else:
                wrapped_inputs = [
                    TypedRef(ref, ti.type)
                    for ref, ti in zip(input_refs, typed_inputs)
                ]
                wrapped_output = TypedOutputRef(
                    output_ref,
                    Type(out_type.st, spec_et, out_type.dt),
                    out_type,
                )
                kernel_fn(*wrapped_inputs, wrapped_output)

        pallas_kwargs = {}
        if tiled:
            pallas_kwargs.update(
                grid=grid, in_specs=in_specs, out_specs=out_specs,
            )
        if compiler_params is not None:
            pallas_kwargs['compiler_params'] = compiler_params
        result_arr = pl.pallas_call(
            jax_kernel,
            out_shape=out_struct,
            interpret=interpret,
            **pallas_kwargs,
        )(*[ti.arr for ti in typed_inputs])

        # The returned TypedJaxArray's Type uses the resolved ExprType (so
        # downstream consumers see the spec / reference, not the kernel's
        # internal expression) and the OutputSpec's ShapeType.
        return TypedJaxArray(
            result_arr,
            Type(out_type.st, spec_et, out_type.dt),
        )

    return runner
