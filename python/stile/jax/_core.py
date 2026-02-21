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

import einops


class TypedJaxArray:
    def __init__(self, arr : jax.Array, type : Type):
        self.arr = arr
        self.type = type

    def slice(self, dim : FullDim, start : int, end : int) -> "TypedJaxArray":
        slice_expr = []
        for i, d in enumerate(self.type.st):
            if dim_contains(d, dim):
                slice_expr.append(slice(start, end))
            else:
                slice_expr.append(slice(None))

        new_type = self.type.slice(dim, start, end)
        return TypedJaxArray(self.arr[tuple(slice_expr)], new_type)

    def repeat(self, dim : Dim) -> "TypedJaxArray":
        nrepeats = dim_size(dim)
        repeated_arr = jnp.repeat(self.arr[None], nrepeats, axis=0)
        new_type = self.type.repeat(dim)
        return TypedJaxArray(repeated_arr, new_type)

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

        new_arr = einops.rearrange(self.arr, f"{lhs_str} -> {rhs_str}")
        new_type = self.type.rearrange(*dims)
        return TypedJaxArray(new_arr, new_type)

    def reduce(self, op : ReduceOpType, dim : Dim) -> "TypedJaxArray":
        new_type = self.type.reduce(op, dim)

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


def _binary_op_helper(
    slf : TypedJaxArray | float,
    other : TypedJaxArray | float,
    op : BinaryOpType,
) -> TypedJaxArray | float:
    lhs_type = slf.type if isinstance(slf, TypedJaxArray) else slf
    rhs_type = other.type if isinstance(other, TypedJaxArray) else other
    new_type = type_from_binary_op(lhs_type, rhs_type, op)

    lhs = slf.arr if isinstance(slf, TypedJaxArray) else slf
    rhs = other.arr if isinstance(other, TypedJaxArray) else other
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


def exp(x : TypedJaxArray) -> TypedJaxArray:
    new_type = t.exp(x.type)
    new_arr = jnp.exp(x.arr)
    return TypedJaxArray(new_arr, new_type)


def sin(x : TypedJaxArray) -> TypedJaxArray:
    new_type = t.sin(x.type)
    new_arr = jnp.sin(x.arr)
    return TypedJaxArray(new_arr, new_type)


def cos(x : TypedJaxArray) -> TypedJaxArray:
    new_type = t.cos(x.type)
    new_arr = jnp.cos(x.arr)
    return TypedJaxArray(new_arr, new_type)


def sqrt(x : TypedJaxArray) -> TypedJaxArray:
    new_type = t.sqrt(x.type)
    new_arr = jnp.sqrt(x.arr)
    return TypedJaxArray(new_arr, new_type)


def maximum(x : TypedJaxArray, y : TypedJaxArray) -> TypedJaxArray:
    return _binary_op_helper(x, y, "max")


def einsum(x : TypedJaxArray, y : TypedJaxArray, einstr : str) -> TypedJaxArray:
    new_arr = einops.einsum(x.arr, y.arr, einstr)
    new_type = t.einsum(x.type, y.type, einstr)
    return TypedJaxArray(new_arr, new_type)


class TypedResult:
    def __init__(self, spec : str):
        self.expected_type = parse_spec_into_type(spec)
        self.shape = tuple(dim_size(d) for d in self.expected_type.st) if self.expected_type.st is not None else tuple()
        self.arr = jnp.zeros(self.shape)

    def assign(self, result : TypedJaxArray):
        if not verify_types_equivalent(
                self.expected_type,
                result.type,
        ):
            raise ValueError(f"Attempted to assign a tensor that does not match the spec! Expected : {self.expected_expr_type}, actual : {result.expr_type}")

        slice_expr = []
        for d in result.type.st:
            ds, de = dim_start(d), dim_end(d)
            slice_expr.append(slice(ds, de))
        self.arr = self.arr.at[tuple(slice_expr)].set(result.arr)


def zeros(shape : tuple[FullDim, ...]) -> TypedJaxArray:
    jax_shape = tuple(dim_size(d) for d in shape)
    arr = jnp.zeros(jax_shape)
    type = Type(
        st=shape,
        et=0.0,
    )
    return TypedJaxArray(arr, type)
