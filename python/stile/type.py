from dataclasses import dataclass
from typing import Literal
from enum import Enum

from .indexing import SymbolicIndex, AffineExpr, to_affine

g_dim_registry : dict[str, "FullDim"] = {}


def as_int(x : SymbolicIndex) -> int | None:
    """
    Return `x` as a plain `int` if it's concretely an integer, else `None`.
    Accepts either a Python int or an `AffineExpr` that happens to have no
    free variables. Use this when logic needs a concrete value; otherwise,
    pass `SymbolicIndex` through symbolically.
    """
    if isinstance(x, int):
        return x
    if isinstance(x, AffineExpr) and not x.terms:
        return x.const
    return None

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
            i.start if i.start is not None else 0,
            i.stop if i.stop is not None else self.size,
        )

@dataclass(frozen=True)
class Sliced:
    dim : "Dim"
    start : SymbolicIndex
    end : SymbolicIndex

Dim = FullDim | Sliced
ShapeType = tuple[Dim, ...]

def dim_start(dim : Dim) -> SymbolicIndex:
    match dim:
        case FullDim(_, _):
            return 0
        case Sliced(d, st, _):
            return dim_start(d) + st

def dim_end(dim : Dim) -> SymbolicIndex:
    match dim:
        case FullDim(_, s):
            return s
        case Sliced(d, _, en):
            return dim_start(d) + en

def dim_name(dim : Dim) -> str:
    match dim:
        case FullDim(n, _):
            return n
        case Sliced(d, _, _):
            return dim_name(d)

def dim_contains(dim : Dim, target : FullDim) -> bool:
    match dim:
        case FullDim(n, s):
            return n == target.name and s == target.size
        case Sliced(d, _, _):
            return dim_contains(d, target)

def dim_size(dim : Dim) -> SymbolicIndex:
    match dim:
        case FullDim(_, s):
            return s
        case Sliced(_, s, e):
            return e - s

def dim_full_dim(dim : Dim) -> FullDim:
    match dim:
        case FullDim(_, _):
            return dim
        case Sliced(d, _, _):
            return dim_full_dim(d)

def simplify_dim(dim : Dim) -> Dim:
    match dim:
        case FullDim(_, _):
            return dim
        case Sliced(d, st, en):
            child = simplify_dim(d)
            match child:
                case FullDim(_, sz):
                    # Collapse Sliced(full, 0, full.size) -> full when we can verify it.
                    st_i, en_i = as_int(st), as_int(en)
                    if st_i == 0 and en_i == sz:
                        return child
                    return Sliced(child, st, en)
                case Sliced(full, child_st, child_en):
                    # Compose nested slices symbolically. When all bounds are
                    # concrete, also sanity-check the range and try to collapse.
                    new_start = child_st + st
                    new_end = child_st + en
                    start_i = as_int(new_start)
                    end_i = as_int(new_end)
                    child_en_i = as_int(child_en)
                    if start_i is not None and end_i is not None and child_en_i is not None:
                        if start_i > child_en_i or end_i > child_en_i:
                            raise ValueError("Invalid slice")
                        if start_i == 0 and end_i == full.size:
                            return full
                    return Sliced(full, new_start, new_end)

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

class DataType(Enum):
    bfloat16 = "bfloat16"
    float32 = "float32"
    float64 = "float64"

dt = DataType

@dataclass(frozen=True)
class Type:
    st : ShapeType
    et : ExprType
    dt : DataType | None = None

    def slice(self, dim : FullDim, start : SymbolicIndex, end : SymbolicIndex) -> "Type":
        dim_type = []
        expr_type = self.et

        dim_found = False
        for i, d in enumerate(self.st):
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

        return Type(tuple(dim_type), self.et, self.dt)

    def repeat(self, dim : Dim) -> "Type":
        new_dim_type = [dim]
        for d in self.st:
            if dim_name(dim) == dim_name(d):
                raise ValueError(f"Cannot repeat a dim that already exists ({dim=})")
            new_dim_type.append(d)

        nrepeats = dim_size(dim)
        new_expr_type = Repeat(dim_full_dim(dim), self.et)
        return Type(tuple(new_dim_type), new_expr_type, self.dt)

    def rearrange(self, *dims : Dim) -> "Type":
        dims = tuple(dim_full_dim(d) for d in dims)

        dims_by_name = {}
        lhs_str = ""
        for d in self.st:
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

        return Type(tuple(new_dim_type), self.et, self.dt)

    def reduce(self, op : ReduceOpType, dim : Dim) -> "Type":
        dim = dim_full_dim(dim)

        new_dim_type = []
        reduction_dim = None
        ireduction_dim = None
        for i, d in enumerate(self.st):
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
        return Type(tuple(new_dim_type), new_expr_type, self.dt)

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
            if slf.st != other.st:
                raise ValueError("Binary operations can only occur between tensors with the same shapes")
            if (
                slf.dt is not None
                and other.dt is not None 
                and slf.dt != other.dt
            ):
                raise ValueError("Binary operations must occur between tensors of the same type! You may have forgotten an explicit cast.")

            new_st = slf.st
            new_et = BinaryOp(
                op=op,
                lhs=slf.et,
                rhs=other.et,
            )
            return Type(new_st, new_et, slf.dt)
        case Type(), x:
            new_st = slf.st
            new_et = BinaryOp(
                op=op,
                lhs=slf.et,
                rhs=Constant(x),
            )
            return Type(new_st, new_et, slf.dt)
        case x, Type():
            new_st = other.st
            new_et = BinaryOp(
                op=op,
                lhs=Constant(x),
                rhs=other.et,
            )
            return Type(new_st, new_et, other.dt)
    assert False
                
def einsum(a : Type, b : Type, einstr : str) -> Type:
    lhs, rhs = einstr.split('->')
    a_str, b_str = lhs.split(',')
    a_dims = a_str.strip().split(' ')
    b_dims = b_str.strip().split(' ')
    rhs_dim_names = rhs.strip().split(' ')

    a_dims_by_name = {}
    b_dims_by_name = {}
    for d in a.st:
        name = dim_name(d)
        if name not in a_dims:
            raise ValueError(f"Dimension {name} not found in tensor A's einsum string")
        a_dims_by_name[name] = d

    for d in b.st:
        name = dim_name(d)
        if name not in b_dims:
            raise ValueError(f"Dimension {name} not found in tensor B's einsum string")
        b_dims_by_name[name] = d

    common_dims = a_dims_by_name.keys() & b_dims_by_name.keys()
    a_repeated = a
    for d in b.st:
        name = dim_name(d)
        if name not in common_dims:
            a_repeated = a_repeated.repeat(d)

    b_repeated = b
    for d in a.st:
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
    new_dim_type = x.st
    new_expr_type = UnaryOp(
        op="exp",
        child=x.et,
    )
    return Type(new_dim_type, new_expr_type, x.dt)

def sin(x : Type) -> Type:
    new_dim_type = x.st
    new_expr_type = UnaryOp(
        op="sin",
        child=x.et,
    )
    return Type(new_dim_type, new_expr_type, x.dt)

def cos(x : Type) -> Type:
    new_dim_type = x.st
    new_expr_type = UnaryOp(
        op="cos",
        child=x.et,
    )
    return Type(new_dim_type, new_expr_type, x.dt)

def sqrt(x : Type) -> Type:
    new_dim_type = x.st,
    new_expr_type = UnaryOp(
        op="sqrt",
        child=x.et,
    )
    return Type(new_dim_type, new_expr_type, x.dt)

def maximum(x : Type | float, y : Type | float) -> Type:
    return type_from_binary_op(x, y, "max")


def override_dims_in_type(type : Type, *dim_override : Dim) -> Type:
    dim_override_by_name = { dim_name(d) : d for d in dim_override }

    def get_overridden(d : Dim):
        name = dim_name(d)
        if name in dim_override_by_name:
            return dim_override_by_name[name]
        return d

    new_st = tuple(get_overridden(d) for d in type.st)
    
    def recursively_replace(et : ExprType) -> ExprType:
        match et:
            case Constant():
                return et
            case Tensor():
                return et
            case UnaryOp(op, child):
                return UnaryOp(
                    op,
                    recursively_replace(child),
                )
            case BinaryOp(op, lhs, rhs):
                return BinaryOp(
                    op,
                    recursively_replace(lhs),
                    recursively_replace(rhs),
                )
            case Repeat(dim, child):
                return Repeat(
                    get_overridden(dim),
                    recursively_replace(child),
                )
            case Reduce(op, dim, child):
                return Reduce(
                    op,
                    get_overridden(dim),
                    recursively_replace(child)
                )
    new_et = recursively_replace(type.et)
    return Type(new_st, new_et, type.dt)
