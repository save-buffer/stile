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

