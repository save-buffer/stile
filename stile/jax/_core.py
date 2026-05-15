try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise ImportError(
        "JAX support requires the jax extra: pip install stile[jax]"
    ) from None

import functools
import inspect
import math

import stile.type as t
from ..type import *
from ..specification import parse_spec_into_type, _parse_predicate, LexState
from ..verification import (
    verify_types_equivalent, verify_exprs_equivalent, verify_dims_equivalent,
    normalize as _normalize, _substitute_lv_in_expr,
    simplify_under_active_loop_scope as _simplify_under_loop_scope,
    substitute_loop_var_in_et,
)
from ..indexing import (
    evaluate as _eval_index, free_vars as _free_vars, to_affine, LoopVariable,
    SymbolicInt, AffineExpr,
    declare_index_properties, declare_block_pairing, tensor_element,
    declare_tensor_boundary, resolve_symbolic_index,
    runtime_scalar_max,
)
from ..tracing import (
    CoverageTracker, _g_runtime_arrs, _bound_as_int, _bound_runtime,
)
from .. import LoopScope, _active_loop_scopes

import einops


# Module-level registry of `LoopVariable` name → JAX value, used by
# tjax internals (notably `_build_predicate_array`) to evaluate symbolic
# slice offsets at runtime. Set by `typed_pallas_call` before tracing
# the kernel — `_pid_<i>` resolves to `pl.program_id(i)`. Outside a
# binding context, symbolic offsets stay an error.
_loop_var_resolver : dict = {}


# Set by `tjax.jit`'s jax.jit-traced body wrapper; positive when we're
# inside a tjax.jit-traced execution. tjax.fori_loop reads this to decide
# whether to dispatch to jax.lax.fori_loop even when bounds are Python
# ints — under @tjax.jit, the body's TypedResult/invariant verification
# already fired on the symbolic pass and this pass is purely for
# numerical execution.
_g_jit_trace_depth = [0]




class loop_var_binding:
    """Context manager that binds `LoopVariable` names to JAX values for
    the duration of the `with` block. Used by `typed_pallas_call` to
    expose `pl.program_id` to in-kernel mask construction."""
    def __init__(self, bindings : dict):
        self.bindings = bindings
        self.previous = None
    def __enter__(self):
        global _loop_var_resolver
        self.previous = _loop_var_resolver
        _loop_var_resolver = self.bindings
        return self
    def __exit__(self, *_):
        global _loop_var_resolver
        _loop_var_resolver = self.previous


def _resolve_to_runtime(symbolic, var_resolver):
    """Convert a `SymbolicIndex` (int / `LoopVariable` / `AffineExpr`)
    to a runtime value — a Python int when fully concrete, a `jnp`
    expression involving the resolver's bound jax-side values otherwise.
    Errors if a `LoopVariable` is encountered that neither the resolver
    nor the atom's `runtime_value` field covers."""
    s_int = as_int(symbolic)
    if s_int is not None:
        return s_int
    aff = to_affine(symbolic)
    result = aff.const
    for var, coeff in aff.terms:
        if var.name in var_resolver:
            result = result + coeff * var_resolver[var.name]
            continue
        # Fallback: a `SymbolicInt` bound via `tjax.jit`'s tracer
        # injection carries its runtime value directly on the atom.
        if var.runtime_value is not None:
            result = result + coeff * var.runtime_value
            continue
        raise ValueError(
            f"Cannot resolve symbolic LoopVariable {var.name!r} to "
            f"a runtime value. Bind it via `loop_var_binding` or "
            f"use a concrete slice offset."
        )
    return result


class TypedJaxArray:
    def __init__(self, arr : jax.Array | None, type : Type):
        # `arr` is None when this TypedJaxArray was produced inside a rolled
        # loop (or any context where slice bounds / reduction sizes aren't
        # concrete ints). Type/verification still propagate normally; the
        # concrete computation is skipped and every downstream op forwards
        # None.
        self.arr = arr
        self.type = type

    # JAX pytree registration: the `arr` is the leaf (so jax.lax.fori_loop
    # and friends can thread the array through their traced graph), the
    # `type` is aux data (Python-static, must match across iterations).
    # Since each tjax op derives its output type deterministically from
    # input types, a verified loop body — whose carry-type is a fixed
    # point — produces matching aux on every iteration.
    def tree_flatten(self):
        return (self.arr,), self.type

    @classmethod
    def tree_unflatten(cls, type_, children):
        (arr,) = children
        return cls(arr, type_)

    def slice(self, dim : FullDim, start : SymbolicIndex, end : SymbolicIndex) -> "TypedJaxArray":
        new_type = self.type.slice(dim, start, end)
        if self.arr is None:
            return TypedJaxArray(None, new_type)
        start_rt = _bound_runtime(start)
        end_rt = _bound_runtime(end)
        if start_rt is None or end_rt is None:
            return TypedJaxArray(None, new_type)
        # Concrete Python ints — fast indexing path.
        if isinstance(start_rt, int) and isinstance(end_rt, int):
            slice_expr = []
            for d in self.type.st:
                if dim_contains(d, dim):
                    slice_expr.append(slice(start_rt, end_rt))
                else:
                    slice_expr.append(slice(None))
            return TypedJaxArray(self.arr[tuple(slice_expr)], new_type)
        # Tracer path: jax.lax.dynamic_slice needs a tracer start but a
        # static slice size. The size = end - start must reduce to a
        # concrete int after symbolic cancellation (the typical
        # `k*BN`/`(k+1)*BN` shape does — terms cancel, const remains).
        size_aff = to_affine(end) - to_affine(start)
        if size_aff.terms:
            raise ValueError(
                f"Slice with tracer bounds requires a statically-known "
                f"slice size; got bounds ({start}, {end}) with non-constant "
                f"difference {size_aff}."
            )
        slice_size = size_aff.const
        starts = []
        sizes = []
        for d in self.type.st:
            if dim_contains(d, dim):
                starts.append(start_rt)
                sizes.append(slice_size)
            else:
                starts.append(0)
                sizes.append(as_int(dim_size(d)))
        new_arr = jax.lax.dynamic_slice(self.arr, tuple(starts), tuple(sizes))
        return TypedJaxArray(new_arr, new_type)

    def repeat(self, dim : Dim) -> "TypedJaxArray":
        new_type = self.type.repeat(dim)
        nrepeats = as_int(dim_size(dim))
        if self.arr is None or nrepeats is None:
            return TypedJaxArray(None, new_type)
        return TypedJaxArray(jnp.repeat(self.arr[None], nrepeats, axis=0), new_type)

    def rearrange(self, *dims : Dim) -> "TypedJaxArray":
        dims = tuple(dim_full_dim(d) for d in dims)

        dims_by_name = {}
        lhs_str = ""
        for d in self.type.st:
            name = dim_name(d)
            dims_by_name[name] = d
            lhs_str += f"{name} "

        rhs_str = ""
        names = [dim_name(d) for d in dims]
        for n in names:
            if n not in dims_by_name:
                raise ValueError(f"Trying to rearrange with unknown dim {n}")
            rhs_str += f"{n} "

        new_type = self.type.rearrange(*dims)
        if self.arr is None:
            return TypedJaxArray(None, new_type)
        return TypedJaxArray(einops.rearrange(self.arr, f"{lhs_str} -> {rhs_str}"), new_type)

    def reduce(self, op : ReduceOpType, dim : Dim) -> "TypedJaxArray":
        new_type = self.type.reduce(op, dim)
        if self.arr is None:
            return TypedJaxArray(None, new_type)

        for i, d in enumerate(self.type.st):
            if dim_name(dim) == dim_name(d):
                ireduction_dim = i
                break
        match op:
            case "sum":
                new_arr = self.arr.sum(axis=ireduction_dim)
            case "max":
                new_arr = self.arr.max(axis=ireduction_dim)
        return TypedJaxArray(new_arr, new_type)

    def sum(self, dim : Dim) -> "TypedJaxArray":
        return self.reduce("sum", dim)

    def max(self, dim : Dim) -> "TypedJaxArray":
        return self.reduce("max", dim)

    def block_at(
        self,
        dim : FullDim,
        offsets : "TypedJaxArray",
        g : "SymbolicIndex",
    ) -> "TypedJaxArray":
        """
        Slice `self` along `dim` to the block `[offsets[g], offsets[g+1])`.
        Wraps `self.slice(dim, tensor_element(offsets, g),
        tensor_element(offsets, g+1))` — the slice bounds are
        `SymbolicInt`s carrying the `(tensor_name, position)` source,
        which the verifier reads to recognize "this slice is exactly
        block g of an offsets tensor" and apply the block-constant
        rewrite to any `gather` of the paired index in the body.
        """
        offsets_name = offsets.type.et.name
        start = tensor_element(offsets_name, to_affine(g))
        end = tensor_element(offsets_name, to_affine(g) + 1)
        new_type = self.type.slice(dim, start, end)
        new_arr = None
        s_int = _bound_as_int(start)
        e_int = _bound_as_int(end)
        if self.arr is not None and s_int is not None and e_int is not None:
            axis = next(
                i for i, d in enumerate(self.type.st)
                if dim_name(d) == dim_name(dim)
            )
            slc = [slice(None)] * len(self.type.st)
            slc[axis] = slice(s_int, e_int)
            new_arr = self.arr[tuple(slc)]
        return TypedJaxArray(new_arr, new_type)

    def gather(self, dim : FullDim, idx : "TypedJaxArray") -> "TypedJaxArray":
        """
        Index into `self` along `dim` using runtime integer tensor
        `idx`. Output shape replaces `dim` with `idx`'s sole dim.
        """
        new_type = self.type.gather(dim, idx.type)
        if self.arr is None or idx.arr is None:
            return TypedJaxArray(None, new_type)
        axis = next(
            i for i, d in enumerate(self.type.st) if dim_name(d) == dim_name(dim)
        )
        return TypedJaxArray(jnp.take(self.arr, idx.arr, axis=axis), new_type)

    def scatter(self, dim : FullDim, idx : "TypedJaxArray") -> "TypedJaxArray":
        """
        Dual of `gather`: write `self` into a zero-initialized output
        whose `dim` axis is populated at positions given by `idx`. The
        output's shape replaces `self`'s `idx`-dim with `dim`. Other
        positions of the output (those not addressed by any `idx[m]`)
        are zero.
        """
        new_type = self.type.scatter(dim, idx.type)
        if self.arr is None or idx.arr is None:
            return TypedJaxArray(None, new_type)
        idx_dim_name = dim_name(idx.type.st[0])
        axis = next(
            i for i, d in enumerate(self.type.st)
            if dim_name(d) == idx_dim_name
        )
        out_shape = list(self.arr.shape)
        out_shape[axis] = as_int(dim_size(dim))
        result = jnp.zeros(tuple(out_shape), dtype=self.arr.dtype)
        # Move the scatter axis to position 0 for `at[].add()` semantics.
        moved = jnp.moveaxis(self.arr, axis, 0)
        out_moved = jnp.zeros(
            (as_int(dim_size(dim)),) + moved.shape[1:], dtype=self.arr.dtype,
        )
        out_moved = out_moved.at[idx.arr].add(moved)
        result = jnp.moveaxis(out_moved, 0, axis)
        return TypedJaxArray(result, new_type)

    def __add__(self, other) -> "TypedJaxArray":
        return _binary_op_helper(self, other, "+")

    def __sub__(self, other) -> "TypedJaxArray":
        return _binary_op_helper(self, other, "-")

    def __mul__(self, other) -> "TypedJaxArray":
        return _binary_op_helper(self, other, "*")

    def __truediv__(self, other) -> "TypedJaxArray":
        return _binary_op_helper(self, other, "/")

    def __radd__(self, other) -> "TypedJaxArray":
        return _binary_op_helper(other, self, "+")

    def __rsub__(self, other) -> "TypedJaxArray":
        return _binary_op_helper(other, self, "-")

    def __rmul__(self, other) -> "TypedJaxArray":
        return _binary_op_helper(other, self, "*")

    def __rtruediv__(self, other) -> "TypedJaxArray":
        return _binary_op_helper(other, self, "/")

    def __matmul__(self, other) -> "TypedJaxArray":
        return einsum(self, other, "M N, N K -> M K")

    def where(self, predicate_str : str) -> "TypedJaxArray":
        """
        Multiplicative-mask sugar: `self * tjax.mask(self.type.st, p)`.
        Same surface shape the spec parser's `where`-clause produces, so
        a kernel calling `.where(p)` and a spec written with `where`
        normalize to the same form. For bias-form masks, build the
        tagged tensor explicitly via `tjax.mask(..., 0.0, -jnp.inf)`
        and add it instead.
        """
        return self * mask(self.type.st, predicate_str)

    def assert_equivalent(self, spec : str, *dim_override : Dim):
        expected_type = parse_spec_into_type(spec)
        expected_type = override_dims_in_type(expected_type, *dim_override)
        are_equivalent = verify_exprs_equivalent(
            expected_type.et,
            self.type.et,
        )
        assert are_equivalent


jax.tree_util.register_pytree_node(
    TypedJaxArray,
    TypedJaxArray.tree_flatten,
    TypedJaxArray.tree_unflatten,
)


def _build_predicate_array(
    domain,
    st : tuple[Dim, ...],
    in_value : float,
    out_value : float,
):
    """
    Materialize a `Domain` (DNF: OR over conjunctions of `expr >= 0`) into
    a `jnp` array of shape matching `st`, with `in_value` at positions
    satisfying the domain and `out_value` elsewhere. Each axis index is
    offset by its `dim_start` so sliced dims evaluate the predicate at
    their absolute positions (a `qctx[8:16]` slice supplies q-values
    8..15, not 0..7). Axes whose dim isn't referenced by the predicate
    are constant along that axis (broadcast).
    """
    shape = tuple(as_int(dim_size(d)) for d in st)
    if any(s is None for s in shape):
        raise ValueError(
            f"Predicate-mask requires concrete dim sizes; got shape={shape}"
        )
    # Slice offsets may be symbolic (an `AffineExpr` over `LoopVariable`s)
    # when this is invoked inside a tiled Pallas kernel. Resolve via the
    # active `loop_var_binding`, which maps each `_pid_<i>` to its
    # `pl.program_id(i)` runtime value. Concrete-int starts pass through
    # unchanged.
    starts = tuple(_resolve_to_runtime(dim_start(d), _loop_var_resolver) for d in st)
    name_to_axis = {dim_name(d) : i for i, d in enumerate(st)}

    final_mask = None
    for conj in domain.disjuncts:
        conj_mask = None
        for c in conj:
            term_value = jnp.full(shape, c.expr.const, dtype=jnp.int32)
            for var, coeff in c.expr.terms:
                if var.name not in name_to_axis:
                    raise ValueError(
                        f"Predicate variable {var.name!r} not in tensor's dims"
                    )
                axis = name_to_axis[var.name]
                idx = jnp.arange(shape[axis], dtype=jnp.int32) + starts[axis]
                shape_bc = [1] * len(shape)
                shape_bc[axis] = shape[axis]
                term_value = term_value + coeff * idx.reshape(shape_bc)
            constraint = term_value >= 0
            conj_mask = constraint if conj_mask is None else (conj_mask & constraint)
        if conj_mask is None:
            conj_mask = jnp.ones(shape, dtype=jnp.bool_)
        final_mask = conj_mask if final_mask is None else (final_mask | conj_mask)
    if final_mask is None:
        final_mask = jnp.zeros(shape, dtype=jnp.bool_)
    return jnp.where(final_mask, in_value, out_value).astype(jnp.float32)


def mask(
    shape : tuple[Dim, ...],
    predicate_str : str,
    in_value : float = 1.0,
    out_value : float = 0.0,
) -> "TypedJaxArray":
    """
    A constant tagged tensor: `in_value` at positions satisfying the
    predicate, `out_value` elsewhere. The type's `st` is `shape` (which
    may carry `Sliced` dims so the mask composes with sliced operands
    via binary ops); the TagCond's domain references the dims' names via
    `LoopVariable`s, so the same predicate is meaningful regardless of
    slicing.

    Common uses:
      - mult-mask (default): `tjax.mask((qctx, nctx), "nctx <= qctx")`
        gives 1 inside / 0 outside; multiply into a tensor.
      - bias-mask: `tjax.mask((qctx, nctx), "nctx <= qctx", 0.0, -jnp.inf)`
        gives 0 inside / -inf outside; add to a tensor (the bias-form
        used by online softmax).

    `.where(predicate)` is sugar for `self * mask(self.type.st, p)`.
    """
    lex = LexState(predicate_str)
    pred_domain = _parse_predicate(lex)
    dim_names_in_shape = {dim_name(d) for d in shape}
    for v in pred_domain.variables:
        if v.name not in dim_names_in_shape:
            raise ValueError(
                f"`mask` predicate references dim {v.name!r} not in "
                f"shape {sorted(dim_names_in_shape)}"
            )
    full_dims = tuple(dim_full_dim(d) for d in shape)
    mask_et = Tensor(
        dims=full_dims,
        tag=TagCond(
            domain=pred_domain,
            if_true=Constant(in_value),
            if_false=Constant(out_value),
        ),
        name="_mask",
    )
    # Inside a symbolic loop-invariant trace the slice offsets reference
    # an unbound `LoopVariable`; we still need the *type* to flow through
    # the kernel for verification, but no actual array is materialized.
    try:
        arr = _build_predicate_array(pred_domain, shape, in_value, out_value)
    except ValueError:
        arr = None
    return TypedJaxArray(arr, Type(shape, mask_et, None))


def _coerce_scalar(x):
    """If x is a scalar jax.Array, convert it to a Python float so it can flow through
    Constant(...) into the normalized expression as a hashable value.
    Inside Pallas kernels, jax tracers can't `.item()` — fall through
    and let downstream code see the raw tracer."""
    if isinstance(x, jax.Array) and x.ndim == 0:
        try:
            return x.item()
        except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
            return x
    return x

def _binary_op_helper(
    slf : TypedJaxArray | float,
    other : TypedJaxArray | float,
    op : BinaryOpType,
) -> TypedJaxArray | float:
    slf = _coerce_scalar(slf)
    other = _coerce_scalar(other)
    lhs_type = slf.type if isinstance(slf, TypedJaxArray) else slf
    rhs_type = other.type if isinstance(other, TypedJaxArray) else other
    new_type = type_from_binary_op(lhs_type, rhs_type, op)

    lhs = slf.arr if isinstance(slf, TypedJaxArray) else slf
    rhs = other.arr if isinstance(other, TypedJaxArray) else other
    if lhs is None or rhs is None:
        return TypedJaxArray(None, new_type)
    match op:
        case "+":
            new_arr = lhs + rhs
        case "-":
            new_arr = lhs - rhs
        case "*":
            new_arr = lhs * rhs
        case "/":
            new_arr = lhs / rhs
        case "max":
            new_arr = jnp.maximum(lhs, rhs)
        case _:
            raise ValueError(f"Unknown op {op}")

    return TypedJaxArray(new_arr, new_type)


def _apply_unary(x : TypedJaxArray, new_type : Type, jnp_fn) -> TypedJaxArray:
    if x.arr is None:
        return TypedJaxArray(None, new_type)
    return TypedJaxArray(jnp_fn(x.arr), new_type)


def exp(x : TypedJaxArray) -> TypedJaxArray:
    return _apply_unary(x, t.exp(x.type), jnp.exp)


def sin(x : TypedJaxArray) -> TypedJaxArray:
    return _apply_unary(x, t.sin(x.type), jnp.sin)


def cos(x : TypedJaxArray) -> TypedJaxArray:
    return _apply_unary(x, t.cos(x.type), jnp.cos)


def sqrt(x):
    """`tjax.sqrt` — eager on Python scalars, JAX-traced on TypedJaxArrays.
    Lets `tjax.sqrt(dhead.size)` (a Python int) return a concrete float
    so it can be used as a divisor inside Pallas kernels (where
    `jnp.sqrt` would produce an abstract tracer that can't be coerced
    to a Python scalar)."""
    if isinstance(x, (int, float)):
        return math.sqrt(x)
    return _sqrt_typed(x)


def _sqrt_typed(x : TypedJaxArray) -> TypedJaxArray:
    return _apply_unary(x, t.sqrt(x.type), jnp.sqrt)


def maximum(x : TypedJaxArray, y : TypedJaxArray) -> TypedJaxArray:
    return _binary_op_helper(x, y, "max")


def einsum(x : TypedJaxArray, y : TypedJaxArray, einstr : str) -> TypedJaxArray:
    new_type = t.einsum(x.type, y.type, einstr)
    if x.arr is None or y.arr is None:
        return TypedJaxArray(None, new_type)
    return TypedJaxArray(einops.einsum(x.arr, y.arr, einstr), new_type)


def fori_loop(lower, upper, body_fn, init_val, *, invariant=None):
    """
    Verification-mode analogue of `jax.lax.fori_loop`. Three paths:

    - **Concrete `lower`/`upper`** (no invariant): fold `body_fn` over
      `range(lower, upper)` with concrete integer indices.

    - **Symbolic `upper`, no invariant**: detect sum (`init=0`) or max
      (`init=-inf`) accumulators and emit a `ParametricReduce`.

    - **Symbolic `upper` with invariant** (Hoare-style): the user
      declares what the loop's state should *be* at iteration `k` as
      a stile spec string referencing the free LoopVariable `k`. For
      tuple state, pass a tuple of strings matching the state shape.
      Verifier discharges base case (`init_val == invariant[k=0]`)
      and inductive step (`body(k, invariant[k]) == invariant[k+1]`);
      returns a typed value with `et = invariant[k=upper]`. Cost is
      invariant to the trip count.

    Signature mirrors `jax.lax.fori_loop`; same user code lowers to
    real `jax.lax.fori_loop` at runtime.
    """
    # Tracer dispatch first: when we're inside a jax-traced execution
    # (typically under `tjax.jit`'s jit-compiled wrapper), defer to
    # `jax.lax.fori_loop`. Two ways this can happen — either a bound
    # is itself a jax tracer (e.g. a `RuntimeScalar` kwarg bound to a
    # jax value), or the carry leaves are tracers (the tensor inputs
    # are jax-traced but the loop bounds are plain Python ints). Both
    # mean: verification already happened on the symbolic pass — this
    # is pure execution, skip the invariant re-check.
    lower_rt = _bound_runtime(lower)
    upper_rt = _bound_runtime(upper)
    bounds_are_tracers = (
        lower_rt is not None and upper_rt is not None
        and (
            isinstance(lower_rt, jax.Array) or isinstance(upper_rt, jax.Array)
        )
    )
    if (
        bounds_are_tracers
        or _carry_has_tracer(init_val)
        or _g_jit_trace_depth[0] > 0
    ):
        if lower_rt is None or not isinstance(lower_rt, (int, jax.Array)):
            lower_rt = as_int(lower)
        if upper_rt is None or not isinstance(upper_rt, (int, jax.Array)):
            upper_rt = as_int(upper)
        return _fori_loop_jax_traced(lower_rt, upper_rt, body_fn, init_val)

    if invariant is not None:
        return _fori_loop_with_invariant(lower, upper, body_fn, init_val, invariant)

    lower_i, upper_i = as_int(lower), as_int(upper)
    if lower_i is not None and upper_i is not None:
        carry = init_val
        for i in range(lower_i, upper_i):
            carry = body_fn(i, carry)
        return carry

    # Symbolic path: dispatch on init_val's identity value.
    #   init=0     -> sum accumulator, body(k, s) = s + f(k)
    #   init=-inf  -> max accumulator, body(k, s) = max(s, f(k))
    # Both are the identity element for their respective op, which lets us
    # extract the per-iteration contribution cleanly.
    scalar_init = _init_scalar(init_val)
    if scalar_init == 0:
        op = "sum"
    elif scalar_init is not None and math.isinf(scalar_init) and scalar_init < 0:
        op = "max"
    else:
        raise NotImplementedError(
            "tjax.fori_loop symbolic path currently supports init=0 (sum "
            "accumulator) or init=-inf (max accumulator). Other init values "
            "need explicit op dispatch or loop-invariant annotations."
        )

    name = f"_fori_{len(_active_loop_scopes)}"
    with LoopScope(name, lower, upper) as k:
        first_iter = body_fn(k, init_val)
        if op == "sum":
            delta = first_iter - init_val
            return init_val + _wrap_parametric_reduce(k, lower, upper, "sum", delta)
        # max: init is -inf (identity), so first_iter = max(-inf, f(k)) = f(k)
        # after normalization — the outer max(-inf, …) collapses away when
        # `first_iter` is normalized inside the ParametricReduce.
        return _wrap_parametric_reduce(k, lower, upper, "max", first_iter)


def _carry_has_tracer(init_val) -> bool:
    """
    True if any leaf of `init_val` carries a jax tracer (rather than a
    concrete jax array or Python value). Tells `fori_loop` that we're
    inside a jax-traced execution — the loop should run via
    `jax.lax.fori_loop`, not the symbolic verification path.
    """
    leaves = init_val if isinstance(init_val, tuple) else (init_val,)
    for leaf in leaves:
        arr = leaf.arr if isinstance(leaf, TypedJaxArray) else leaf
        if isinstance(arr, jax.core.Tracer):
            return True
    return False


def _fori_loop_jax_traced(lower, upper, body_fn, init_val):
    """
    Execute `body_fn` via `jax.lax.fori_loop` when one of the bounds is
    a jax tracer (typically because `tjax.jit` is running the user
    function under `jax.jit` with a `RuntimeScalar` bound to a jax
    value). The loop variable `i` is wrapped in a
    `SymbolicInt(runtime_value=i)` so tjax ops inside the body see the
    same atom shape they would during symbolic verification — their
    `_bound_runtime` lookups then resolve to the tracer, taking the
    `jax.lax.dynamic_slice`-style execution paths.

    Only jax arrays flow through `jax.lax.fori_loop`'s carry — types
    are stripped on the way in and re-applied on the way out, holding
    the input type fixed across iterations. The body's *computed*
    output type would otherwise grow per iter (each iteration adds
    another layer of `BinaryOp` etc. to the carry's `et`), which
    `jax.lax.fori_loop` would reject as a pytree-aux mismatch.
    Verification has already proved the carry is a fixed point of the
    body via the invariant, so dropping the per-iter type is safe.

    Init values that arrive as Python scalars (or unshaped jax scalars)
    are broadcast to the carry's actual shape, which we discover by
    running the body once symbolically — `tjax.maximum(-jnp.inf,
    tile_max)` etc. produces a `(qctx,)`-shaped output where the init
    was `()`, and `jax.lax.fori_loop` rejects shape mismatches between
    input and output of the body.

    Verification is *not* run on this path — it happened once already,
    on `tjax.jit`'s first call. This pass is purely for runtime
    execution.
    """
    iter_var_name = f"__fori_i_{len(_active_loop_scopes)}__"
    is_tuple = isinstance(init_val, tuple)
    init_leaves = list(init_val) if is_tuple else [init_val]

    # Discover the carry's shape by running the body once symbolically
    # (arr=None everywhere), using a free SymbolicInt for the iter var.
    # The output types tell us each leaf's expected shape — we use that
    # to broadcast scalar init values for the real run.
    sym_state_leaves = [
        TypedJaxArray(None, leaf.type) if isinstance(leaf, TypedJaxArray)
        else leaf
        for leaf in init_leaves
    ]
    sym_state = tuple(sym_state_leaves) if is_tuple else sym_state_leaves[0]
    i_sym_probe = SymbolicInt(name=iter_var_name)
    sym_result = body_fn(i_sym_probe, sym_state)
    sym_result_leaves = list(sym_result) if is_tuple else [sym_result]

    init_types = []
    init_arrs = []
    for leaf, sym_leaf in zip(init_leaves, sym_result_leaves):
        if isinstance(sym_leaf, TypedJaxArray):
            target_type = sym_leaf.type
            target_shape = tuple(as_int(dim_size(d)) for d in target_type.st)
        else:
            target_type = None
            target_shape = ()
        init_types.append(target_type)
        if isinstance(leaf, TypedJaxArray):
            arr = leaf.arr
        else:
            arr = jnp.asarray(leaf)
        if arr.shape != target_shape:
            arr = jnp.broadcast_to(arr, target_shape)
        init_arrs.append(arr)

    def jax_body(i, carry_arrs):
        i_sym = SymbolicInt(name=iter_var_name, runtime_value=i)
        wrapped = [
            TypedJaxArray(arr, t_) if t_ is not None else arr
            for arr, t_ in zip(carry_arrs, init_types)
        ]
        state = tuple(wrapped) if is_tuple else wrapped[0]
        new_state = body_fn(i_sym, state)
        new_leaves = list(new_state) if is_tuple else [new_state]
        return tuple(
            leaf.arr if isinstance(leaf, TypedJaxArray) else jnp.asarray(leaf)
            for leaf in new_leaves
        )

    final_arrs = jax.lax.fori_loop(lower, upper, jax_body, tuple(init_arrs))
    final_leaves = [
        TypedJaxArray(arr, t_) if t_ is not None else arr
        for arr, t_ in zip(final_arrs, init_types)
    ]
    return tuple(final_leaves) if is_tuple else final_leaves[0]


def jit(spec : str):
    """
    Verify-once, run-many decorator. Takes a spec string for the
    function's return value: on the first call, the function is run
    symbolically (any kwarg whose name matches a declared
    `runtime_scalar` is substituted with a bare `SymbolicInt`) and the
    returned `TypedJaxArray`'s type is checked against `spec`. If the
    types are equivalent, the function is `jax.jit`-compiled for fast
    execution on subsequent calls — each call re-binds the scalar
    kwargs as `SymbolicInt(name, runtime_value=tracer)` so the body's
    tjax ops resolve them to live tracers via `_bound_runtime`.

    Per-tile verification (`TypedResult.assign`/`done`,
    `fori_loop(..., invariant=...)`, `assert_equivalent`) still
    happens *inside* the function — the decorator's spec verification
    catches the kernel's whole-function output type, and the
    in-function checks handle the structural pieces that compose to
    it. Both halves are required for full verification, just like
    `TypedResult.done()` is required after `assign` calls.

    Cache key: the set of `RuntimeScalar`-named kwargs. Each distinct
    set re-verifies (analogous to `jax.jit` re-tracing on different
    abstract input signatures).
    """
    def decorate(fn):
        sig = inspect.signature(fn)
        cache = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Any kwarg whose NAME matches a declared `runtime_scalar`
            # is a RuntimeScalar-typed arg — we swap it for a
            # `SymbolicInt` at verification time and re-wrap it with
            # `runtime_value=tracer` at execution time.
            scalar_names = tuple(sorted(
                name for name in bound.arguments
                if runtime_scalar_max(name) is not None
            ))
            sig_key = scalar_names

            if sig_key not in cache:
                # Verification pass: run with symbolic scalars.
                verify_args = dict(bound.arguments)
                for name in scalar_names:
                    verify_args[name] = SymbolicInt(name)
                result = fn(**verify_args)
                if not isinstance(result, TypedJaxArray):
                    raise TypeError(
                        f"@tjax.jit'd function must return a TypedJaxArray "
                        f"to verify against its spec; got {type(result).__name__}."
                    )
                expected = parse_spec_into_type(
                    spec, loop_vars=set(scalar_names),
                )
                if not verify_exprs_equivalent(result.type.et, expected.et):
                    raise AssertionError(
                        f"@tjax.jit return type does not match spec.\n"
                        f"  spec: {expected.et}\n"
                        f"  actual: {result.type.et}"
                    )

                # Compile a jax.jit wrapper that re-injects each scalar
                # as `SymbolicInt(name, runtime_value=tracer)` and marks
                # the trace as a tjax.jit-execution pass — so
                # tjax.fori_loop and friends know to run via jax.lax
                # primitives instead of re-running verification.
                def jit_body(tensor_kwargs, scalar_kwargs):
                    runtime_args = dict(tensor_kwargs)
                    for name, tracer in scalar_kwargs.items():
                        runtime_args[name] = SymbolicInt(
                            name, runtime_value=tracer,
                        )
                    _g_jit_trace_depth[0] += 1
                    try:
                        return fn(**runtime_args)
                    finally:
                        _g_jit_trace_depth[0] -= 1

                cache[sig_key] = jax.jit(jit_body)

            compiled = cache[sig_key]
            scalar_kwargs = {
                name: bound.arguments[name] for name in scalar_names
            }
            tensor_kwargs = {
                name: val for name, val in bound.arguments.items()
                if name not in scalar_names
            }
            return compiled(tensor_kwargs, scalar_kwargs)

        return wrapper
    return decorate


def _fori_loop_with_invariant(lower, upper, body_fn, init_val, invariant):
    """
    Hoare-style verified fori_loop. `invariant` is either a single spec
    string (scalar state) or a list/tuple of spec strings matching
    `init_val`'s structure. Discharges base case and inductive step;
    returns a typed value (or tuple) with `et = invariant[k=upper]`.
    """
    is_tuple_state = isinstance(invariant, (tuple, list))
    inv_strs = list(invariant) if is_tuple_state else [invariant]
    state_leaves = list(init_val) if is_tuple_state else [init_val]
    if len(inv_strs) != len(state_leaves):
        raise ValueError(
            f"`invariant` has {len(inv_strs)} elements but init_val has "
            f"{len(state_leaves)}; they must match."
        )

    # Each entry is either a spec string (parsed with `k` plus any
    # free `LoopVariable`s in the loop bounds as free vars) or a
    # pre-built `Type` whose `et` already references those vars
    # directly. Pre-built types let invariants incorporate `Gather`
    # and other ExprTypes the surface spec language doesn't have
    # syntax for yet. Extracting free vars from `lower`/`upper` lets
    # the loop range itself be symbolic — e.g. a runtime
    # `n_used_pages` for paged-decode early-exit.
    bound_free_vars = (_free_vars(lower) | _free_vars(upper))
    loop_vars_for_spec = {"k"} | {v.name for v in bound_free_vars}
    inv_types = [
        s if isinstance(s, Type)
        else parse_spec_into_type(s, loop_vars=loop_vars_for_spec)
        for s in inv_strs
    ]
    k_var = LoopVariable("k")

    # 1. Base case: init_val == invariant[k=0].
    for inv_t, init_leaf in zip(inv_types, state_leaves):
        inv_at_0 = _substitute_lv_in_expr(_normalize(inv_t.et), k_var, 0)
        init_norm = _normalize_state_leaf(init_leaf)
        if init_norm != inv_at_0:
            raise AssertionError(
                f"Loop invariant base case failed: init_val ({init_leaf}) "
                f"does not normalize to invariant[k=0]."
            )

    # 2. Inductive step: build a typed state from each invariant evaluated
    # at symbolic `k`, run the body, check result == invariant[k+1].
    # LoopScope's name doubles as its `LoopVariable`'s identity — keep it
    # `"k"` so the verifier's `active_loop_vars()` agrees with the
    # `LoopVariable("k")` that's free in the parsed invariants.
    with LoopScope("k", lower, upper):
        symbolic_state_leaves = [
            TypedJaxArray(None, inv_t) for inv_t in inv_types
        ]
        symbolic_state = (
            tuple(symbolic_state_leaves) if is_tuple_state
            else symbolic_state_leaves[0]
        )
        next_state = body_fn(k_var, symbolic_state)
        next_leaves = list(next_state) if is_tuple_state else [next_state]
        if len(next_leaves) != len(inv_types):
            raise AssertionError(
                "Loop body returned a state with a different shape than "
                "the invariant."
            )
        for idx, (inv_t, next_leaf) in enumerate(zip(inv_types, next_leaves)):
            inv_at_kplus1 = _simplify_under_loop_scope(_substitute_lv_in_expr(
                _normalize(inv_t.et), k_var, to_affine(k_var) + 1,
            ))
            if not isinstance(next_leaf, TypedJaxArray):
                raise TypeError(
                    f"Loop body must return TypedJaxArrays for invariant "
                    f"verification; got {type(next_leaf).__name__}."
                )
            next_norm = _simplify_under_loop_scope(_normalize(next_leaf.type.et))
            if next_norm != inv_at_kplus1:
                raise AssertionError(
                    f"Loop invariant inductive step failed at state index {idx}: "
                    "body(k, invariant[k]) does not normalize to "
                    "invariant[k+1]."
                )

    # 3. Return value is invariant evaluated at k = upper, materialized
    # at the ExprType level so downstream tjax ops can compose normally.
    final_leaves = []
    for inv_t in inv_types:
        et_at_upper = substitute_loop_var_in_et(inv_t.et, k_var, upper)
        final_leaves.append(
            TypedJaxArray(None, Type(inv_t.st, et_at_upper, inv_t.dt))
        )
    return tuple(final_leaves) if is_tuple_state else final_leaves[0]


def _normalize_state_leaf(leaf):
    """Normalize a state-leaf value (Python scalar, jax.Array, or
    TypedJaxArray) for comparison against an invariant at k=0."""
    if isinstance(leaf, TypedJaxArray):
        return _normalize(leaf.type.et)
    if isinstance(leaf, (int, float)):
        return _normalize(t.Constant(float(leaf)))
    if isinstance(leaf, jax.Array) and leaf.ndim == 0:
        try:
            return _normalize(t.Constant(float(leaf.item())))
        except Exception:
            pass
    raise TypeError(
        f"Cannot normalize state leaf of type {type(leaf).__name__} for "
        f"invariant base-case verification."
    )


def _init_scalar(init_val):
    """Extract a Python scalar value from init_val if it's scalar-typed."""
    if isinstance(init_val, (int, float)):
        return init_val
    if isinstance(init_val, jax.Array) and init_val.ndim == 0:
        return init_val.item()
    return None


def _wrap_parametric_reduce(
    loop_var,
    lo : SymbolicIndex,
    hi : SymbolicIndex,
    op : ReduceOpType,
    body : TypedJaxArray,
) -> TypedJaxArray:
    """
    Package `body` as a `ParametricReduce` in the `ExprType` layer.
    """
    new_et = ParametricReduce(loop_var, lo, hi, op, body.type.et)
    new_type = Type(body.type.st, new_et, body.type.dt)
    return TypedJaxArray(None, new_type)


class TypedResult(CoverageTracker):
    """
    jax binding of `CoverageTracker`: backs the coverage buffer with a
    `jnp.ndarray` and uses `at[].set()` for the per-tile write. All
    verification logic — per-tile ET equivalence, the `_assigned`
    ledger, `done()`'s coverage check — lives on the base class and is
    framework-agnostic.
    """
    def _init_array(self, shape):
        return jnp.zeros(shape)

    def _write_to_arr(self, slices, tile_arr):
        self.arr = self.arr.at[slices].set(tile_arr)


def zeros(shape : tuple[FullDim, ...]) -> TypedJaxArray:
    jax_shape = tuple(dim_size(d) for d in shape)
    arr = jnp.zeros(jax_shape)
    type = Type(
        st=shape,
        et=t.Constant(0.0),
    )
    return TypedJaxArray(arr, type)


def runtime_index(
    name : str,
    dim : FullDim,
    *,
    values_in : FullDim | None = None,
    arr : "jnp.ndarray | None" = None,
    permutation : bool = False,
    partition : bool = False,
    block_sorted_paired_with : str | None = None,
    boundary_values : "dict[int, int] | None" = None,
) -> TypedJaxArray:
    """
    A 1-d integer-valued tensor named `name`, of shape `(dim,)`, used
    as a runtime index into another tensor's `values_in` dim via
    `source.gather(values_in, idx)`. `values_in` is recorded for
    documentation / future property-aware simplifications but isn't
    enforced — the verifier treats the gather opaquely. If `arr` is
    omitted, a default identity-like array is generated; supply a real
    `arr` for runtime correctness.

    `permutation=True` declares the index is a bijection from `dim` to
    `values_in` — same size, every value in `[0, values_in.size)`
    hit exactly once. Unlocks the round-trip rewrite
    `scatter(gather(Y, perm), perm) = Y`.

    `partition=True` declares each input maps to exactly one value in
    `[0, values_in.size)` (a function, not necessarily injective).
    Unlocks the partition-sum collapse for MoE-style kernels.
    """
    props = []
    if permutation:
        props.append("permutation")
    if partition:
        props.append("partition")
    if props:
        declare_index_properties(name, *props)
    if block_sorted_paired_with is not None:
        declare_block_pairing(block_sorted_paired_with, name)
    if boundary_values is not None:
        for position, value in boundary_values.items():
            declare_tensor_boundary(name, position, value)
    type = Type(
        st=(dim,),
        et=t.Tensor(dims=(dim,), name=name),
    )
    if arr is None:
        size = as_int(dim_size(dim))
        arr = jnp.arange(size) if size is not None else None
    if arr is not None:
        _g_runtime_arrs[name] = arr
    return TypedJaxArray(arr, type)
