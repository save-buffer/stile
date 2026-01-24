from dataclasses import dataclass
from typing import Literal
from enum import Enum

g_dim_registry : dict[str, "FullDim"] = {}

@dataclass(frozen=True)
class FullDim:
    name : str
    size : int

    def __post_init__(self):
        if self.name in g_dim_registry:
            if g_dim_registry[self.name] != self:
                raise ValueError(f"Attempted to redefine a dimension with a different size (old={g_dim_registry[self.name].size}, new={self.size})!")
        else:
            g_dim_registry[self.name] = self

    def __getitem__(self, i : slice) -> "Sliced":
        return Sliced(
            self,
            i.start,
            i.stop,
        )

@dataclass(frozen=True)
class Sliced:
    dim : "Dim"
    start : int
    end : int

Dim = FullDim | Sliced
DimType = tuple[Dim, ...]

def dim_start(dim : Dim):
    match dim: 
        case FullDim(_, _):
            return 0
        case Sliced(d, st, _):
            return dim_start(d) + st

def dim_end(dim : Dim):
    match dim:
        case FullDim(_, s):
            return s
        case Sliced(d, _, en):
            return dim_start(d) + en

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
                    if st == 0 and en == sz:
                        return child
                    return Sliced(child, st, en)
                case Sliced(full, child_st, child_en):
                    length = en - st
                    slice_start = child_st + st
                    slice_end = child_st + st + length
                    if slice_start > child_en or slice_end > child_en:
                        raise ValueError("Invalid slice")
                    if child_st + st == 0 and child_st + st + length == full.size:
                        return full
                    return Sliced(full, child_st + st, child_st + st + length)

@dataclass(frozen=True)
class Constant:
    value: float

@dataclass(frozen=True)
class Tensor:
    dims : tuple[FullDim, ...]

UnaryOpType = Literal["exp", "sin", "cos", "sqrt"]

@dataclass(frozen=True)
class UnaryOp:
    op : UnaryOpType
    child : "ExprType"

BinaryOpType = Literal["+", "-", "*", "/", "max"]

@dataclass(frozen=True)
class BinaryOp:
    op : BinaryOpType
    lhs : "ExprType"
    rhs : "ExprType"

@dataclass(frozen=True)
class Repeat:
    dim : Dim
    child : "ExprType"

ReduceOpType = Literal["sum", "max"]

@dataclass(frozen=True)
class Reduce:
    op : ReduceOpType
    dim : Dim
    child : "ExprType"

ExprType = Constant | Tensor | UnaryOp | BinaryOp | Repeat | Reduce

@dataclass(frozen=True)
class Type:
    dt : DimType
    et : ExprType

    def slice(self, dim : FullDim, start : int, end : int) -> "Type":
        dim_type = []
        expr_type = self.et

        dim_found = False
        for i, d in enumerate(self.dt):
            if dim_contains(d, dim):
                dim_type.append(
                    Sliced(
                        d,
                        start,
                        end,
                    )
                )
                dim_found = True
            else:
                dim_type.append(d)

        if not dim_found:
            raise ValueError(f"Invalid dim {dim}")

        for i in range(len(dim_type)):
            dim_type[i] = simplify_dim(dim_type[i])

        return Type(tuple(dim_type), self.et)

    def repeat(self, dim : Dim) -> "Type":
        new_dim_type = [dim]
        for d in self.dt:
            if dim_name(dim) == dim_name(d):
                raise ValueError(f"Cannot repeat a dim that already exists ({dim=})")
            new_dim_type.append(d)

        nrepeats = dim_size(dim)
        new_expr_type = Repeat(dim_full_dim(dim), self.et)
        return Type(tuple(new_dim_type), new_expr_type)

    def rearrange(self, *dims : Dim) -> "Type":
        dims = tuple(dim_full_dim(d) for d in dims)

        dims_by_name = {}
        lhs_str = ""
        for d in self.dt:
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

        return Type(tuple(new_dim_type), self.et)
    
    def reduce(self, op : ReduceOpType, dim : Dim) -> "Type":
        dim = dim_full_dim(dim)

        new_dim_type = []
        reduction_dim = None
        ireduction_dim = None
        for i, d in enumerate(self.dt):
            if dim_name(dim) != dim_name(d):
                new_dim_type.append(d)
            else:
                ireduction_dim = i
                reduction_dim = d
        if reduction_dim is None:
            raise ValueError(f"Unknown reduction dimension {d}")
        assert ireduction_dim is not None
        
        new_expr_type = Reduce(
            op=op,
            dim=reduction_dim,
            child=self.et
        )
        return Type(tuple(new_dim_type), new_expr_type)
    
    def sum(self, dim : Dim) -> "Type":
        return self.reduce("sum", dim)

    def max(self, dim : Dim) -> "Type":
        return self.reduce("max", dim)

    def __add__(self, other) -> "Type":
        return type_from_binary_op(self, other, "+")

    def __sub__(self, other) -> "Type":
        return type_from_binary_op(self, other, "-")

    def __mul__(self, other) -> "Type":
        return type_from_binary_op(self, other, "*")

    def __truediv__(self, other) -> "Type":
        return type_from_binary_op(self, other, "/")

    def __radd__(self, other) -> "Type":
        return type_from_binary_op(other, self, "+")

    def __rsub__(self, other) -> "Type":
        return type_from_binary_op(other, self, "-")

    def __rmul__(self, other) -> "Type":
        return type_from_binary_op(other, self, "*")

    def __rtruediv__(self, other) -> "Type":
        return type_from_binary_op(other, self, "/")

    def __matmul__(self, other) -> "Type":
        return einsum(self, other, "M N, N K -> M K")
    
def type_from_binary_op(slf : Type | float, other : Type | float, op : BinaryOpType) -> Type:
    match slf, other:
        case Type(), Type():
            if slf.dt != other.dt:
                raise ValueError("Binary operations can only occur between tensors with the same shapes")
            new_dt = slf.dt
            new_et = BinaryOp(
                op=op,
                lhs=slf.et,
                rhs=other.et,
            )
            return Type(new_dt, new_et)
        case Type(), x:
            new_dt = slf.dt
            new_et = BinaryOp(
                op=op,
                lhs=slf.et,
                rhs=Constant(x),
            )
            return Type(new_dt, new_et)
        case x, Type():
            new_dt = other.dt
            new_et = BinaryOp(
                op=op,
                lhs=Constant(x),
                rhs=other.et,
            )
            return Type(new_dt, new_et)
    assert False
                
def einsum(a : Type, b : Type, einstr : str) -> Type:
    lhs, rhs = einstr.split('->')
    a_str, b_str = lhs.split(',')
    a_dims = a_str.strip().split(' ')
    b_dims = b_str.strip().split(' ')
    rhs_dim_names = rhs.strip().split(' ')

    a_dims_by_name = {}
    b_dims_by_name = {}
    for d in a.dt:
        name = dim_name(d)
        if name not in a_dims:
            raise ValueError(f"Dimension {name} not found in tensor A's einsum string")
        a_dims_by_name[name] = d

    for d in b.dt:
        name = dim_name(d)
        if name not in b_dims:
            raise ValueError(f"Dimension {name} not found in tensor B's einsum string")
        b_dims_by_name[name] = d

    common_dims = a_dims_by_name.keys() & b_dims_by_name.keys()
    a_repeated = a
    for d in b.dt:
        name = dim_name(d)
        if name not in common_dims:
            a_repeated = a_repeated.repeat(d)

    b_repeated = b
    for d in a.dt:
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
        c = c.sum(d)

    return c

def exp(x : Type) -> Type:
    new_dim_type = x.dt
    new_expr_type = UnaryOp(
        op="exp",
        child=x.et,
    )
    return Type(new_dim_type, new_expr_type)

def sin(x : Type) -> Type:
    new_dim_type = x.dt
    new_expr_type = UnaryOp(
        op="sin",
        child=x.et,
    )
    return Type(new_dim_type, new_expr_type)

def cos(x : Type) -> Type:
    new_dim_type = x.dt
    new_expr_type = UnaryOp(
        op="cos",
        child=x.et,
    )
    return Type(new_dim_type, new_expr_type)
    
def sqrt(x : Type) -> Type:
    new_dim_type = x.dt,
    new_expr_type = UnaryOp(
        op="sqrt",
        child=x.et,
    )
    return Type(new_dim_type, new_expr_type)

def maximum(x : Type | float, y : Type | float) -> Type:
    return type_from_binary_op(x, y, "max")

