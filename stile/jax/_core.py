try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise ImportError(
        "JAX support requires the jax extra: pip install stile[jax]"
    ) from None

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
)
from .. import LoopScope, _active_loop_scopes

import einops


# Module-level registry of `LoopVariable` name → JAX value, used by
# tjax internals (notably `_build_predicate_array`) to evaluate symbolic
# slice offsets at runtime. Set by `typed_pallas_call` before tracing
# the kernel — `_pid_<i>` resolves to `pl.program_id(i)`. Outside a
# binding context, symbolic offsets stay an error.
_loop_var_resolver : dict = {}


# Module-level registry of runtime index/tensor name → JAX array.
# `runtime_index` registers under the index's name so per-iter helpers
# (e.g. `_resolve_te_to_int`) can look up `offsets[g]` as a concrete
# Python int when both the array and `g` are concrete, enabling the
# carry-pattern runtime path: `assign`'s `at[].set()` write fires even
# when bounds came from a `tensor_element` lookup.
_g_runtime_arrs : dict = {}


def _resolve_te_to_int(x : "SymbolicIndex") -> "int | None":
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


def _bound_as_int(x : "SymbolicIndex") -> "int | None":
    """`as_int` plus `tensor_element` resolution. Used by `assign` to
    decide whether a slice bound can be materialized to a Python int
    for the runtime `at[].set()` write."""
    v = as_int(x)
    if v is not None:
        return v
    return _resolve_te_to_int(x)


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
    Errors if a `LoopVariable` is encountered that the resolver doesn't
    cover."""
    s_int = as_int(symbolic)
    if s_int is not None:
        return s_int
    aff = to_affine(symbolic)
    result = aff.const
    for var, coeff in aff.terms:
        if var.name not in var_resolver:
            raise ValueError(
                f"Cannot resolve symbolic LoopVariable {var.name!r} to "
                f"a runtime value. Bind it via `loop_var_binding` or "
                f"use a concrete slice offset."
            )
        result = result + coeff * var_resolver[var.name]
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

    def slice(self, dim : FullDim, start : SymbolicIndex, end : SymbolicIndex) -> "TypedJaxArray":
        new_type = self.type.slice(dim, start, end)
        start_i, end_i = as_int(start), as_int(end)
        if self.arr is None or start_i is None or end_i is None:
            return TypedJaxArray(None, new_type)
        slice_expr = []
        for d in self.type.st:
            if dim_contains(d, dim):
                slice_expr.append(slice(start_i, end_i))
            else:
                slice_expr.append(slice(None))
        return TypedJaxArray(self.arr[tuple(slice_expr)], new_type)

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


class TypedResult:
    def __init__(self, spec : str):
        self.expected_type = parse_spec_into_type(spec)
        shape_dims = self.expected_type.st if self.expected_type.st is not None else ()
        self.shape = tuple(as_int(dim_size(d)) or 0 for d in shape_dims)
        self.arr = jnp.zeros(self.shape)
        # Each entry is (per-dim bounds tuple, snapshot of active LoopScopes at
        # assign-time). `done()` uses the scope snapshot to unroll symbolic
        # bounds into concrete intervals for coverage checking.
        self._assigned : list[tuple[
            tuple[tuple[SymbolicIndex, SymbolicIndex], ...],
            tuple[LoopScope, ...],
        ]] = []

    def assign(self, result : TypedJaxArray):
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

        # Only copy data when the bounds and the source array are concrete.
        # Bounds that came from `tensor_element(offsets, g)` resolve via the
        # registered runtime array when both `g` and `offsets`'s arr are
        # concrete — that's the path the carry pattern relies on.
        concrete_bounds = tuple(
            (_bound_as_int(s), _bound_as_int(e)) for s, e in per_dim_bounds
        )
        all_concrete_bounds = all(
            s is not None and e is not None for s, e in concrete_bounds
        )
        if result.arr is not None and all_concrete_bounds:
            slice_expr = tuple(slice(s, e) for s, e in concrete_bounds)
            self.arr = self.arr.at[slice_expr].set(result.arr)

    def write_block(self, tile : TypedJaxArray) -> "TypedResult":
        """
        Carry-pattern analog of `assign`: verifies the tile against the
        spec restricted to the tile's slice, records its bounds for
        coverage, and returns `self` so the `TypedResult` can be
        threaded as the `fori_loop` carry. The tile's own type already
        encodes the block (via `Sliced` dims from `block_at`), so no
        explicit (dim, offsets, g) args are needed.
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
        d : "Dim",
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
        # Resolve boundary substitutions and concretize where possible.
        resolved = [
            (resolve_symbolic_index(s), resolve_symbolic_index(e))
            for s, e in intervals
        ]
        # Sort by start. Use as_int for sortable key; symbolic starts
        # sort by their `repr` so the order is deterministic.
        def sort_key(iv):
            s_int = as_int(iv[0])
            return (0, s_int) if s_int is not None else (1, repr(iv[0]))
        resolved.sort(key=sort_key)
        # Sweep: each interval's start must equal the prior end (or 0
        # for the first), and the last interval's end must equal full_size.
        cursor : "SymbolicIndex" = 0
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
            full = dim_full_dim(d)
            spec_et = t.Reduce(op="sum", dim=d, child=spec_et)
            tile_et = t.Reduce(op="sum", dim=d, child=tile_et)
    return verify_exprs_equivalent(spec_et, tile_et)


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
    relevant_vars = _free_vars(start_aff) | _free_vars(end_aff)
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
    relevant_vars = _free_vars(start_aff) | _free_vars(end_aff)
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

    return [
        (_eval_index(start_aff, b), _eval_index(end_aff, b))
        for b in bindings_list
    ]


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
