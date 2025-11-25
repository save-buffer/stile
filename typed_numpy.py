from dataclasses import dataclass
from typing import Literal

import numpy as np
import einops

from type_nodes import *
from egraph import Egraph, EclassID

class Typed:
    def __init__(self, arr : np.ndarray, *dim_type : Dim, expr_type : ExprType | None = None):
        self.arr = arr
        if len(dim_type) != len(self.arr.shape):
            raise ValueError("Number of attributes must match physical dimension")
        self.dim_type : tuple[Dim, ...] = dim_type
        self.expr_type : ExprType = (
            expr_type
            if expr_type is not None
            else Tensor(tuple(dim_full_dim(d) for d in self.dim_type))
        )

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
        new_expr_type = Repeat(dim, self.expr_type)
        return Typed(repeated_arr, *new_dim_type, expr_type=new_expr_type)

    def rearrange(self, *dims : Dim) -> "Typed":
        dims = tuple(dim_full_dim(d) for d in dims)

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

        new_expr_type = Reduce(reduction_dim, self.expr_type)
        new_arr = einops.einsum(self.arr, f"{lhs_str} -> {rhs_str}")
        return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)

    def binary_op(self, other, op : BinaryOpType) -> "Typed":
        return _binary_op_helper(self, other, op)

    def __add__(self, other) -> "Typed":
        return self.binary_op(other, "+")

    def __sub__(self, other) -> "Typed":
        return self.binary_op(other, "-")

    def __mul__(self, other) -> "Typed":
        return self.binary_op(other, "*")

    def __div__(self, other) -> "Typed":
        return self.binary_op(other, "/")

    def __matmul__(self, other) -> "Typed":
        return einsum(self, other, "M N, N K -> M K")

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

def _binary_op_helper(slf, other, op):
    match other:
        case Typed():
            if slf.dim_type != other.dim_type:
                raise ValueError("Binary operations can only occur between tensors with the same shapes")
            match op:
                case "+":
                    new_arr = slf.arr + other.arr
                case "-":
                    new_arr = slf.arr - other.arr
                case "*":
                    new_arr = slf.arr * other.arr
                case "/":
                    new_arr = slf.arr / other.arr

            new_dim_type = slf.dim_type
            new_expr_type = BinaryOp(
                op=op,
                lhs=slf.expr_type,
                rhs=other.expr_type,
            )
            return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)
        case x:
            match op:
                case "+":
                    new_arr = slf.arr + x
                case "-":
                    new_arr = slf.arr - x
                case "*":
                    new_arr = slf.arr * x
                case "/":
                    new_arr = slf.arr / x

            new_dim_type = slf.dim_type
            new_expr_type = BinaryOp(
                op=op,
                lhs=slf.expr_type,
                rhs=Constant(x),
            )
            return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)

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
            a_repeated = a_repeated.repeat(d)

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

@dataclass
class LexState:
    spec : str

    def peek(self) -> str | None:
        return self.spec[0] if self.spec else None

    def consume_whitespace(self):
        self.spec = self.spec.strip()

    def consume(self) -> str | None:
        self.consume_whitespace()
        result = self.peek()
        if result:
            self.spec = self.spec[1:]
        return result

    def expect(self, s : str):
        self.consume_whitespace()
        if not self.spec.startswith(s):
            raise ValueError(f"{s} expected")
        self.spec = self.spec[len(s):]

def infer_dims_from_expr(expr : ExprType) -> tuple[FullDim, ...] | None:
    match expr:
        case Constant(x):
            return None
        case Tensor(dims):
            return dims
        case BinaryOp(_, lhs, rhs):
            lhs_dims = infer_dims_from_expr(lhs)
            rhs_dims = infer_dims_from_expr(rhs)
            if lhs_dims is None:
                return rhs_dims
            if rhs_dims is None:
                return lhs_dims
            if sorted(lhs_dims, key=dim_name) != sorted(rhs_dims, key=dim_name):
                raise ValueError("Invalid spec: binary op between tensors of incompatible shapes")
            return lhs_dims
        case Repeat(along, child):
            dims = infer_dims_from_expr(child)
            assert dims is not None
            return (dim_full_dim(along), *dims)
        case Reduce(along, child):
            dims = infer_dims_from_expr(child)
            assert dims is not None
            along_full = dim_full_dim(along)
            return tuple(d for d in dims if d != along_full)


def _construct_expr_from_einsum(a : ExprType, b : ExprType, rhs : Tensor) -> ExprType:
    dims_a = infer_dims_from_expr(a)
    dims_b = infer_dims_from_expr(b)
    dims_rhs = rhs.dims

    assert dims_a is not None
    assert dims_b is not None

    a_dims_by_name = { dim_name(d) : d for d in dims_a }
    b_dims_by_name = { dim_name(d) : d for d in dims_b }
    rhs_dim_names = { dim_name(d) for d in rhs.dims }

    common_dims = a_dims_by_name.keys() & b_dims_by_name.keys()
    a_repeated, b_repeated = a, b
    for name, d in b_dims_by_name.items():
        if name not in common_dims:
            a_repeated = Repeat(d, a_repeated)

    for name, d in a_dims_by_name.items():
        if name not in common_dims:
            b_repeated = Repeat(d, b_repeated)
    a, b = a_repeated, b_repeated 
    
    reduction_dims = [a_dims_by_name[name] for name in common_dims if name not in rhs_dim_names]

    c = BinaryOp(
        op="*", 
        lhs=a,
        rhs=b,
    )
    for d in reduction_dims:
        c = Reduce(d, c)

    return c

def _parse_dim(lex : LexState) -> FullDim:
    lex.consume_whitespace()
    i = 0
    while i < len(lex.spec) and lex.spec[i].isalpha():
        i += 1

    dim_name = lex.spec[:i]
    lex.spec = lex.spec[i:]
    if dim_name not in g_dim_registry:
        raise ValueError(f"Parsed dim {dim_name} is not a known dimension!")
    return g_dim_registry[dim_name]

def _parse_tensor(lex : LexState) -> Tensor:
    dims = []
    while True:
        dims.append(_parse_dim(lex))
        lex.consume_whitespace()
        nxt = lex.peek()
        if nxt is None or not nxt.isalpha():
            return Tensor(tuple(dims))

def _parse_number(lex : LexState) -> Constant:
    lex.consume_whitespace()
    i = 0
    while i < len(lex.spec) and (lex.spec[i].isdigit() or lex.spec[i] == '.'):
        i += 1
    
    constant = float(lex.spec[:i])
    lex.spec = lex.spec[i:]
    return Constant(constant)

def _parse_factor(lex : LexState) -> ExprType:
    lex.consume_whitespace()
    if lex.peek() == "(":
        lex.consume()
        child = _parse_spec(lex)
        match lex.peek():
            case ",":
                lex.consume()
                einsum_second = _parse_spec(lex)
                lex.expect("->")
                tensor = _parse_tensor(lex)
                lex.expect(")")
                return _construct_expr_from_einsum(child, einsum_second, tensor)
            case ")":
                lex.consume()
                return child
            case bad:
                raise ValueError(f"Invalid character {bad}. Expected einsum string or closing paren")
    else:
        nxt = lex.peek()
        if nxt is None:
            raise ValueError("End of input while parsing expression")
        if nxt.isdigit():
            return _parse_number(lex)
        return _parse_tensor(lex)

def _parse_term(lex : LexState) -> ExprType:
    child = _parse_factor(lex)
    lex.consume_whitespace()
    match lex.peek():
        case "*":
            lex.consume()
            return BinaryOp(
                op="*",
                lhs=child,
                rhs=_parse_term(lex),
            )
        case "/":
            lex.consume()
            return BinaryOp(
                op="/",
                lhs=child,
                rhs=_parse_term(lex),
            )
        case _:
            return child

def _parse_spec(lex : LexState) -> ExprType:
    child = _parse_term(lex)
    lex.consume_whitespace()
    match lex.peek():
        case "+":
            lex.consume()
            return BinaryOp(
                op="+",
                lhs=child,
                rhs=_parse_spec(lex),
            )
        case "-":
            if lex.spec.startswith("->"):
                return child

            lex.consume()
            return BinaryOp(
                op="-",
                lhs=child,
                rhs=_parse_spec(lex),
            )
        case _:
            return child

def parse_spec_into_expr_type(spec : str) -> ExprType:
    """
    Grammar:
    Spec -> Term + Spec | Term - Spec | Term
    Term -> Factor * Term | Factor / Term | Factor
    Factor -> ( Spec ) | ( Spec , Spec -> Tensor ) | Tensor | Number
    Tensor -> Dim " " Tensor | Dim
    """
    lex = LexState(spec)
    return _parse_spec(lex)

def remap_dims_by_name(dims_by_name : dict[str, Dim], expr : ExprType) -> ExprType:
    match expr:
        case Constant(_):
            return expr
        case Tensor(_):
            return expr
        case BinaryOp(op, lhs, rhs):
            new_lhs = remap_dims_by_name(dims_by_name, lhs)
            new_rhs = remap_dims_by_name(dims_by_name, rhs)
            return BinaryOp(
                op,
                new_lhs,
                new_rhs,
            )
        case Repeat(d, child):
            name = dim_name(d)
            new_dim = dims_by_name[name] if name in dims_by_name else d
            new_child = remap_dims_by_name(dims_by_name, child)
            return Repeat(new_dim, new_child)
        case Reduce(d, child):
            name = dim_name(d)
            new_dim = dims_by_name[name] if name in dims_by_name else d
            new_child = remap_dims_by_name(dims_by_name, child)
            return Reduce(new_dim, new_child)

def map_expr_to_dim_type(dim_type : tuple[Dim, ...], expr : ExprType) -> ExprType:
    dims_by_name = { dim_name(d) : d for d in dim_type }
    return remap_dims_by_name(dims_by_name, expr)

# TODO: This simplify_expression is SUPER brittle and only has enough functionality to enable example.py
def simplify_expression(expr : ExprType) -> ExprType:
    if isinstance(expr, BinaryOp) and expr.op == "+":
        # The key is that you're only allowed to add reductions of identical subexpressions,
        # so that means that if you add Reduce[0:5] + Reduce[5:10], you can simplify this to Reduce[0:10]
        if isinstance(expr.lhs, Reduce) and isinstance(expr.rhs, Reduce):
            assert dim_full_dim(expr.lhs.dim) == dim_full_dim(expr.rhs.dim)
            if dim_end(expr.lhs.dim) == dim_start(expr.rhs.dim):
                combined_dim = simplify_dim(
                    Sliced(
                        dim_full_dim(expr.lhs.dim),
                        dim_start(expr.lhs.dim),
                        dim_end(expr.rhs.dim)
                    )
                )
                remap_dims_by_name({ dim_name(combined_dim) : combined_dim }, expr.lhs.child)
                return Reduce(combined_dim, expr.lhs.child)
    return expr

def expr_types_are_equivalent(dim_type : tuple[Dim, ...], expected : ExprType, actual : ExprType) -> bool:
    expected = map_expr_to_dim_type(dim_type, expected)

    egraph = Egraph()
    expected_id = egraph.insert_expression(expected)
    actual_id = egraph.insert_expression(actual)

    egraph.apply_rewrites()
    return egraph.equivalent(expected_id, actual_id)

class TypedResult:
    def __init__(self, spec : str):
        self.expected_expr_type = parse_spec_into_expr_type(spec)

        self.expected_dim_type = infer_dims_from_expr(self.expected_expr_type)
        self.shape = tuple(dim_size(d) for d in self.expected_dim_type) if self.expected_dim_type is not None else tuple()
        self.arr = np.zeros(self.shape)
        
    def assign(self, result : Typed):
        if not expr_types_are_equivalent(
                dim_type=result.dim_type,
                expected=self.expected_expr_type,
                actual=result.expr_type
        ):
            raise ValueError(f"Attempted to assign a tensor that does not match the spec! Expected: {self.expected_expr_type}, actual: {result.expr_type}")

        slice_expr = []
        for d in result.dim_type:
            ds, de = dim_start(d), dim_end(d)
            slice_expr.append(slice(ds, de))
        self.arr[*slice_expr] = result.arr
