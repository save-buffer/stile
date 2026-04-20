try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise ImportError(
        "JAX support requires the jax extra: pip install stile[jax]"
    ) from None

import stile.type as t
from ..type import *
from ..specification import parse_spec_into_type
from ..verification import verify_types_equivalent, verify_exprs_equivalent
from ..indexing import evaluate as _eval_index, free_vars as _free_vars, to_affine
from .. import LoopScope, _active_loop_scopes

import einops


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

    def assert_equivalent(self, spec : str, *dim_override : Dim):
        expected_type = parse_spec_into_type(spec)
        expected_type = override_dims_in_type(expected_type, *dim_override)
        are_equivalent = verify_exprs_equivalent(
            expected_type.et,
            self.type.et,
        )
        assert are_equivalent


def _coerce_scalar(x):
    """If x is a scalar jax.Array, convert it to a Python float so it can flow through
    Constant(...) into the normalized expression as a hashable value."""
    if isinstance(x, jax.Array) and x.ndim == 0:
        return x.item()
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


def sqrt(x : TypedJaxArray) -> TypedJaxArray:
    return _apply_unary(x, t.sqrt(x.type), jnp.sqrt)


def maximum(x : TypedJaxArray, y : TypedJaxArray) -> TypedJaxArray:
    return _binary_op_helper(x, y, "max")


def einsum(x : TypedJaxArray, y : TypedJaxArray, einstr : str) -> TypedJaxArray:
    new_type = t.einsum(x.type, y.type, einstr)
    if x.arr is None or y.arr is None:
        return TypedJaxArray(None, new_type)
    return TypedJaxArray(einops.einsum(x.arr, y.arr, einstr), new_type)


def fori_loop(lower, upper, body_fn, init_val):
    """
    Verification-mode analogue of `jax.lax.fori_loop`. Folds `body_fn` over
    `range(lower, upper)` with concrete integer indices, producing the
    actual final carry as if the loop had been manually unrolled. The
    normalizer then canonicalizes the resulting expression, which for
    online-softmax-style bodies collapses into a closed form that matches
    the full-sequence spec.

    Signature mirrors `jax.lax.fori_loop(lower, upper, body_fn, init_val)`:
    migrating existing code is a one-line import change. When lowered for
    actual execution, the same user code becomes `jax.lax.fori_loop`; only
    the verification pathway unrolls.

    Loop bounds must be concrete ints. Fully dynamic loop counts (e.g.,
    paged attention with variable page count) will need parametric
    reductions in the IR or explicit loop invariants — neither of which
    is built yet.
    """
    lower_i, upper_i = as_int(lower), as_int(upper)
    if lower_i is None or upper_i is None:
        raise NotImplementedError(
            "tjax.fori_loop currently requires concrete integer bounds. "
            "Symbolic loop counts (paged-attention-style dynamic iteration) "
            "will need parametric reductions or explicit loop invariants."
        )
    carry = init_val
    for i in range(lower_i, upper_i):
        carry = body_fn(i, carry)
    return carry


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
        if not verify_types_equivalent(
                self.expected_type,
                result.type,
        ):
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
        all_concrete_bounds = all(
            as_int(s) is not None and as_int(e) is not None
            for s, e in per_dim_bounds
        )
        if result.arr is not None and all_concrete_bounds:
            slice_expr = tuple(slice(as_int(s), as_int(e)) for s, e in per_dim_bounds)
            self.arr = self.arr.at[slice_expr].set(result.arr)

    def done(self):
        """
        Verify that the recorded `assign` calls exactly tile the full output
        shape along every dimension — no gaps, no overlaps. Symbolic bounds
        (from assigns inside a rolled loop) are unrolled over the recorded
        loop scopes to produce concrete intervals before the contiguity
        check.
        """
        for dim_idx, d in enumerate(self.expected_type.st):
            full_size = as_int(dim_size(d))
            if full_size is None:
                continue
            intervals : list[tuple[int, int]] = []
            for bounds, scopes in self._assigned:
                s, e = bounds[dim_idx]
                intervals.extend(_unroll_interval(s, e, scopes))
            if not intervals:
                continue
            # Dedupe: repeat assigns with identical bounds (common for dims
            # that aren't tiled — every tile covers the full extent) are fine.
            intervals = sorted(set(intervals))
            cursor = 0
            for s, e in intervals:
                if s != cursor:
                    raise ValueError(
                        f"Dimension {dim_name(d)!r} has a gap or overlap: "
                        f"cursor at {cursor}, next interval starts at {s}"
                    )
                cursor = e
            if cursor != full_size:
                raise ValueError(
                    f"Dimension {dim_name(d)!r} only covered up to {cursor} "
                    f"of {full_size}"
                )


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
        et=0.0,
    )
    return TypedJaxArray(arr, type)
