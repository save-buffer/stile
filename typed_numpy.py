from dataclasses import dataclass
from typing import Literal

import numpy as np
import einops

a = np.random.randn(10, 10)
b = np.random.randn(10, 10)

# TODO: Prevent creating multiple dims with the same name
@dataclass
class FullDim:
    name : str
    size : int

@dataclass
class Sliced:
    dim : FullDim
    start : int
    end : int

Dim = FullDim | Sliced

def dim_name(dim : Dim):
    match dim:
        case FullDim(n, _):
            return n
        case Sliced(d, _, _):
            return dim_name(d)

def dim_contains(dim : Dim, target : FullDim):
    match dim:
        case FullDim(n, s):
            return n == target.name and s == target.size
        case Sliced(d, _, _):
            return dim_contains(d, target)

def dim_size(dim : Dim) -> int:
    match dim:
        case FullDim(_, s):
            return s
        case Sliced(d, s, e):
            return e - s

def dim_full_dim(dim : Dim) -> FullDim:
    match dim:
        case FullDim(_, _):
            return dim
        case Sliced(d, _, _):
            return dim_full_dim(d)

def simplify_dim(dim : Dim) -> Dim:
    match dim:
        case FullDim(n, sz):
            return dim
        case Sliced(d, st, en):
            child = simplify_dim(d)
            match child:
                case FullDim(n, sz):
                    return Sliced(child, st, en)
                case Sliced(full, child_st, child_en):
                    # TODO: Check bounds
                    length = en - st
                    return Sliced(full, child_st + st, child_st + st + length)

BinaryOpType = Literal["+", "-", "*", "/"]

@dataclass
class BinaryOp:
    lhs : "ExprType"
    rhs : "ExprType"
    op : BinaryOpType

@dataclass
class Repeat:
    child : "ExprType"
    dim : Dim

@dataclass
class Reduce:
    child : "ExprType"
    dim : Dim

ExprType = BinaryOp | Repeat | Reduce | BinaryOp

class Typed:
    def __init__(self, arr : np.ndarray, *dim_type : Dim, expr_type=None):
        self.arr = arr
        if len(dim_type) != len(self.arr.shape):
            raise ValueError("Number of attributes must match physical dimension")
        self.dim_type = dim_type
        self.expr_type = expr_type

    def slice(self, dim : FullDim, start : int | None, end : int | None) -> "Typed":
        slice_expr = []
        dim_type = []
        expr_type = self.expr_type

        st : int = start if start is not None else 0

        dim_found = False
        for i, d in enumerate(self.dim_type):
            if dim_contains(d, dim):
                slice_expr.append(slice(start, end))

                dim_type.append(
                    Sliced(
                        d,
                        st,
                        end if end is not None else self.arr.shape[i] - st,
                    )
                )
                dim_found = True
            else:
                slice_expr.append(slice(None))
                dim_type.append(d)

        if not dim_found:
            raise ValueError(f"Invalid dim {dim}")

        for i in range(len(dim_type)):
            dim_type[i] = simplify_dim(dim_type[i])

        return Typed(self.arr[*slice_expr], *dim_type, expr_type=expr_type)

    def repeat(self, dim : Dim) -> "Typed":
        new_dim_type = [dim]
        for d in self.dim_type:
            if dim_name(dim) == dim_name(d):
                raise ValueError(f"Cannot repeat a dim that already exists ({dim=})")
            new_dim_type.append(d)

        nrepeats = dim_size(dim)
        repeated_arr = self.arr[None, :, :].repeat(nrepeats, axis=0)
        new_expr_type = Repeat(self.expr_type, dim)
        return Typed(repeated_arr, *new_dim_type, expr_type=new_expr_type)

    def rearrange(self, *dims : Dim) -> "Typed":
        dims = [dim_full_dim(d) for d in dims]

        dims_by_name = {}
        lhs_str = ""
        for d in self.dim_type:
            name = dim_name(d)
            dims_by_name[name] = d
            lhs_str += f"{name} "

        new_dim_type = []
        rhs_str = ""
        names = [dim_name(d) for d in dims]
        for n in names:
            if n not in dims_by_name:
                raise ValueError(f"Trying to rearrange with unknown dim {n}")
            new_dim_type.append(dims_by_name[n])
            rhs_str += f"{n} "

        new_arr = einops.rearrange(self.arr, f"{lhs_str} -> {rhs_str}")
        return Typed(new_arr, *new_dim_type, expr_type=self.expr_type)

    def reduce(self, dim : Dim) -> "Typed":
        dim = dim_full_dim(dim)

        new_dim_type = []
        reduction_dim = None
        lhs_str = ""
        rhs_str = ""
        for d in self.dim_type:
            name = dim_name(d) + " "
            lhs_str += name
            if dim_name(dim) != dim_name(d):
                new_dim_type.append(d)
                rhs_str += name
            else:
                reduction_dim = d
        if reduction_dim is None:
            raise ValueError(f"Unknown reduction dimension {d}")

        new_expr_type = Reduce(self.expr_type, reduction_dim)
        new_arr = einops.einsum(self.arr, f"{lhs_str} -> {rhs_str}")
        return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)

    def binary_op(self, other : "Typed", op : BinaryOpType):
        if self.dim_type != other.dim_type:
            raise ValueError("Binary operations can only occur between tensors with the same shapes")
        match op:
            case "+":
                new_arr = self.arr + other.arr
            case "-":
                new_arr = self.arr - other.arr
            case "*":
                new_arr = self.arr * other.arr
            case "/":
                new_arr = self.arr / other.arr

        new_dim_type = self.dim_type
        new_expr_type = BinaryOp(self.expr_type, other.expr_type, "*")
        return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)

    def __add__(self, other : "Typed") -> "Typed":
        return self.binary_op(other, "+")

    def __sub__(self, other : "Typed") -> "Typed":
        return self.binary_op(other, "-")

    def __mul__(self, other : "Typed") -> "Typed":
        return self.binary_op(other, "*")

    def __div__(self, other : "Typed") -> "Typed":
        return self.binary_op(other, "/")

    def set(self, other : "Typed"):
        if self.dim_type != other.dim_type:
            raise ValueError("Only assignment between tiles of the same dim type is allowed")
        self.arr = other.arr
        self.expr_type = other.expr_type

    @property
    def type(self):
        return self.dim_type

    @property
    def shape(self):
        return self.arr.shape

def einsum(a : Typed, b : Typed, einstr : str) -> Typed:
    lhs, rhs = einstr.split('->')
    a_str, b_str = lhs.split(',')
    a_dims = a_str.strip().split(' ')
    b_dims = b_str.strip().split(' ')
    rhs_dim_names = rhs.strip().split(' ')

    a_dims_by_name = {}
    b_dims_by_name = {}
    for d in a.dim_type:
        name = dim_name(d)
        if name not in a_dims:
            raise ValueError(f"Dimension {name} not found in tensor A's einsum string")
        a_dims_by_name[name] = d

    for d in b.dim_type:
        name = dim_name(d)
        if name not in b_dims:
            raise ValueError(f"Dimension {name} not found in tensor B's einsum string")
        b_dims_by_name[name] = d

    common_dims = a_dims_by_name.keys() & b_dims_by_name.keys()
    a_repeated = a
    for d in b.dim_type:
        name = dim_name(d)
        if name not in common_dims:
            a_repeated = a.repeat(d)

    b_repeated = b
    for d in a.dim_type:
        name = dim_name(d)
        if name not in common_dims:
            b_repeated = b_repeated.repeat(d)

    a, b = a_repeated, b_repeated

    reduction_dims = []
    for d in common_dims:
        if d not in rhs_dim_names:
            if a_dims_by_name[d] != b_dims_by_name[d]:
                raise ValueError(f"Trying to reduce along mismatching slices of dimension {d}: A={a_dims_by_name[d]}, B={b_dims_by_name[d]}")
            reduction_dims.append(a_dims_by_name[d])

    rhs_dims = []
    for d in rhs_dim_names:
        if d in a_dims_by_name:
            rhs_dims.append(a_dims_by_name[d])
        elif d in b_dims_by_name:
            rhs_dims.append(b_dims_by_name[d])

    a = a.rearrange(*rhs_dims, *reduction_dims)
    b = b.rearrange(*rhs_dims, *reduction_dims)

    c = a * b

    for d in reduction_dims:
        c = c.reduce(d)

    return c
 
    result_dim_type = []
    for d in rhs_dims:
        assert d in a_dims_by_name or d in b_dims_by_name
        if d in a_dims_by_name and d in b_dims_by_name:
            if a_dims_by_name[d] != b_dims_by_name[d]:
                raise ValueError(f"Batch dimension of einsum is operating on mismatching slices! A={a_dims_by_name[d]}, B={b_dims_by_name[d]}")
            result_dim_type.append(a_dims_by_name[d])
        elif d in a_dims_by_name:
            result_dim_type.append(a_dims_by_name[d])
        elif d in b_dims_by_name:
            result_dim_type.append(b_dims_by_name[d])

    result = einops.einsum(a.arr, b.arr, einstr)

M, N, K = FullDim('M', 10), FullDim('N', 10), FullDim('K', 10)

a = Typed(a, M, N)
b = Typed(b, N, K)

a_sliced = a.slice(M, 0, 5).slice(N, 0, 5)
b_sliced = b.slice(N, 0, 5).slice(K, 0, 5)

print("Shape:", a_sliced.shape)
print("Type: ", a_sliced.type)

c_tile0 = einsum(a_sliced, b_sliced, "M N, N K -> M K")

print("Shape:    ", c_tile0.shape)
print("DimType:  ", c_tile0.type)
print("ExprType: ", c_tile0.expr_type)

c = np.zeros((10, 10))
c = Typed(c, M, K)

tile_size = 5
for im in range(0, 10, tile_size):
    for ik in range(0, 10, tile_size):
        c_sliced = c.slice(M, im, im + tile_size).slice(K, ik, ik + tile_size)
        c_accum = None
        for in_ in range(0, 10, tile_size):
            tile_a = a.slice(M, im, im + tile_size).slice(N, in_, in_ + tile_size)
            tile_b = b.slice(N, in_, in_ + tile_size).slice(K, ik, ik + tile_size)
            tile_c = einsum(tile_a, tile_b, "M N, N K -> M K")
            c_accum = c_accum + tile_c if c_accum is not None else tile_c
        c_sliced.set(c_accum)
        print(f"{im=}, {ik=}, {c_sliced.expr_type}")
        
