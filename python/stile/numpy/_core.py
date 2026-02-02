import stile.type as t
from ..type import *
from ..specification import parse_spec_into_type
from ..verification import verify_types_equivalent, verify_exprs_equivalent

import numpy as np
import einops


class TypedNumpyArray:
    def __init__(self, arr : np.ndarray, type : Type):
        self.arr = arr
        self.type = type

    def slice(self, dim : FullDim, start : int, end : int) -> "TypedNumpyArray":
        slice_expr = []
        for i, d in enumerate(self.type.dt):
            if dim_contains(d, dim):
                slice_expr.append(slice(start, end))
            else:
                slice_expr.append(slice(None))

        new_type = self.type.slice(dim, start, end)
        return TypedNumpyArray(self.arr[*slice_expr], new_type)

    def repeat(self, dim : Dim) -> "TypedNumpyArray":
        nrepeats = dim_size(dim)
        repeated_arr = self.arr[None].repeat(nrepeats, axis=0)
        new_type = self.type.repeat(dim)
        return TypedNumpyArray(repeated_arr, new_type)

    def rearrange(self, *dims : Dim) -> "TypedNumpyArray":
        dims = tuple(dim_full_dim(d) for d in dims)

        dims_by_name = {}
        lhs_str = ""
        for d in self.type.dt:
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
        return TypedNumpyArray(new_arr, new_type)

    def reduce(self, op : ReduceOpType, dim : Dim) -> "TypedNumpyArray":
        new_type = self.type.reduce(op, dim)

        for i, d in enumerate(self.type.dt):
            if dim_name(dim) == dim_name(d):
                ireduction_dim = i
                break
        match op:
            case "sum":
                new_arr = self.arr.sum(axis=ireduction_dim)
            case "max":
                new_arr = self.arr.max(axis=ireduction_dim)
        return TypedNumpyArray(new_arr, new_type)

    def sum(self, dim : Dim) -> "TypedNumpyArray":
        return self.reduce("sum", dim)

    def max(self, dim : Dim) -> "TypedNumpyArray":
        return self.reduce("max", dim)

    def __add__(self, other) -> "TypedNumpyArray":
        return _binary_op_helper(self, other, "+")

    def __sub__(self, other) -> "TypedNumpyArray":
        return _binary_op_helper(self, other, "-")

    def __mul__(self, other) -> "TypedNumpyArray":
        return _binary_op_helper(self, other, "*")

    def __truediv__(self, other) -> "TypedNumpyArray":
        return _binary_op_helper(self, other, "/")

    def __radd__(self, other) -> "TypedNumpyArray":
        return _binary_op_helper(other, self, "+")

    def __rsub__(self, other) -> "TypedNumpyArray":
        return _binary_op_helper(other, self, "-")

    def __rmul__(self, other) -> "TypedNumpyArray":
        return _binary_op_helper(other, self, "*")

    def __rtruediv__(self, other) -> "TypedNumpyArray":
        return _binary_op_helper(other, self, "/")

    def __matmul__(self, other) -> "TypedNumpyArray":
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
    slf : TypedNumpyArray | float,
    other : TypedNumpyArray | float,
    op : BinaryOpType,
) -> TypedNumpyArray | float:
    lhs_type = slf.type if isinstance(slf, TypedNumpyArray) else slf
    rhs_type = other.type if isinstance(other, TypedNumpyArray) else other
    new_type = type_from_binary_op(lhs_type, rhs_type, op)

    lhs = slf.arr if isinstance(slf, TypedNumpyArray) else slf
    rhs = other.arr if isinstance(other, TypedNumpyArray) else other
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
            new_arr = np.maximum(lhs, rhs)
        case _:
            raise ValueError(f"Unknown op {op}")

    return TypedNumpyArray(new_arr, new_type)


def exp(x : TypedNumpyArray) -> TypedNumpyArray:
    new_type = t.exp(x.type)
    new_arr = np.exp(x.arr)
    return TypedNumpyArray(new_arr, new_type)


def sin(x : TypedNumpyArray) -> TypedNumpyArray:
    new_type = t.sin(x.type)
    new_arr = np.sin(x.arr)
    return TypedNumpyArray(new_arr, new_type)


def cos(x : TypedNumpyArray) -> TypedNumpyArray:
    new_type = t.cos(x.type)
    new_arr = np.cos(x.arr)
    return TypedNumpyArray(new_arr, new_type)


def sqrt(x : TypedNumpyArray) -> TypedNumpyArray:
    new_type = t.sqrt(x.type)
    new_arr = np.sqrt(x.arr)
    return TypedNumpyArray(new_arr, new_type)


def maximum(x : TypedNumpyArray, y : TypedNumpyArray) -> TypedNumpyArray:
    return _binary_op_helper(x, y, "max")


def einsum(x : TypedNumpyArray, y : TypedNumpyArray, einstr : str) -> TypedNumpyArray:
    new_arr = einops.einsum(x.arr, y.arr, einstr)
    new_type = t.einsum(x.type, y.type, einstr)
    return TypedNumpyArray(new_arr, new_type)


class TypedResult:
    def __init__(self, spec : str):
        self.expected_type = parse_spec_into_type(spec)
        self.shape = tuple(dim_size(d) for d in self.expected_type.dt) if self.expected_type.dt is not None else tuple()
        self.arr = np.zeros(self.shape)

    def assign(self, result : TypedNumpyArray):
        if not verify_types_equivalent(
                self.expected_type,
                result.type,
        ):
            raise ValueError(f"Attempted to assign a tensor that does not match the spec! Expected: {self.expected_expr_type}, actual: {result.expr_type}")

        slice_expr = []
        for d in result.type.dt:
            ds, de = dim_start(d), dim_end(d)
            slice_expr.append(slice(ds, de))
        self.arr[*slice_expr] = result.arr


def zeros(shape : tuple[FullDim, ...]) -> TypedNumpyArray:
    np_shape = tuple(dim_size(d) for d in shape)
    arr = np.zeros(np_shape)
    type = Type(
        dt=shape,
        et=0.0,
    )
    return TypedNumpyArray(arr, type)
