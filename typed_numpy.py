import dataclasses
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
        self.dim_type : DimType = dim_type
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
        repeated_arr = self.arr[None].repeat(nrepeats, axis=0)
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

    def reduce(self, op : ReduceOpType, dim : Dim) -> "Typed":
        dim = dim_full_dim(dim)

        new_dim_type = []
        reduction_dim = None
        ireduction_dim = None
        for i, d in enumerate(self.dim_type):
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
            child=self.expr_type
        )
        match op:
            case "sum":
                new_arr = self.arr.sum(axis=ireduction_dim)
            case "max":
                new_arr = self.arr.max(axis=ireduction_dim)

        return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)

    def sum(self, dim : Dim) -> "Typed":
        return self.reduce("sum", dim)

    def max(self, dim : Dim) -> "Typed":
        return self.reduce("max", dim)

    def binary_op(self, other, op : BinaryOpType) -> "Typed":
        return _binary_op_helper(self, other, op)

    def __add__(self, other) -> "Typed":
        return self.binary_op(other, "+")

    def __sub__(self, other) -> "Typed":
        return self.binary_op(other, "-")

    def __mul__(self, other) -> "Typed":
        return self.binary_op(other, "*")

    def __truediv__(self, other) -> "Typed":
        return self.binary_op(other, "/")

    def __fadd__(self, other) -> "Typed":
        return self.binary_op(other, "+")

    def __rsub__(self, other) -> "Typed":
        return self.binary_op(other, "-")

    def __rmul__(self, other) -> "Typed":
        return self.binary_op(other, "*")

    def __rtruediv__(self, other) -> "Typed":
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
                case "max":
                    new_arr = np.maximum(slf.arr, other.arr)

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
                case "max":
                    new_arr = np.maximum(slf.arr, x)

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
        c = c.sum(d)

    return c

def exp(x : Typed) -> Typed:
    new_dim_type = x.dim_type
    new_expr_type = UnaryOp(
        op="exp",
        child=x.expr_type,
    )
    new_arr = np.exp(x.arr)
    return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)

def sin(x : Typed) -> Typed:
    new_dim_type = x.dim_type
    new_expr_type = UnaryOp(
        op="sin",
        child=x.expr_type,
    )
    new_arr = np.sin(x.arr)
    return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)

def cos(x : Typed) -> Typed:
    new_dim_type = x.dim_type
    new_expr_type = UnaryOp(
        op="cos",
        child=x.expr_type,
    )
    new_arr = np.cos(x.arr)
    return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)
    
def sqrt(x : Typed) -> Typed:
    new_dim_type = x.dim_type
    new_expr_type = UnaryOp(
        op="sqrt",
        child=x.expr_type,
    )
    new_arr = np.sqrt(x.arr)
    return Typed(new_arr, *new_dim_type, expr_type=new_expr_type)

def max(x : Typed, y : Typed) -> Typed:
    return _binary_op_helper(x, y, "max")

@dataclass
class LexState:
    spec : str

    def consume_whitespace(self):
        self.spec = self.spec.strip()

    def peek(self) -> str | None:
        self.consume_whitespace()
        return self.spec[0] if self.spec else None

    def maybe_consume(self, *args) -> str | None:
        self.consume_whitespace()
        for a in args:
            if self.spec.startswith(a):
                self.spec = self.spec[len(a):]
                return a
        return None

    def startswith(self, s):
        return self.spec.startswith(s)

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
        case UnaryOp(_, child):
            return infer_dims_from_expr(child)
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
        case Reduce(_, along, child):
            dims = infer_dims_from_expr(child)
            assert dims is not None
            along_full = dim_full_dim(along)
            return tuple(d for d in dims if d != along_full)

def _construct_binary_reduction_expr(
    a_type : tuple[DimType, ExprType],
    b_type : tuple[DimType, ExprType],
    rhs : DimType,
    reduction : ReduceOpType = "sum",
) -> ExprType:
    dims_a, expr_a = a_type
    dims_b, expr_b = b_type
    a_dims_by_name = { dim_name(d) : d for d in dims_a }
    b_dims_by_name = { dim_name(d) : d for d in dims_b }
    rhs_dim_names = { dim_name(d) for d in rhs }

    common_dims = a_dims_by_name.keys() & b_dims_by_name.keys()
    a_repeated, b_repeated = expr_a, expr_b
    for name, d in b_dims_by_name.items():
        if name not in common_dims:
            a_repeated = Repeat(d, a_repeated)

    for name, d in a_dims_by_name.items():
        if name not in common_dims:
            b_repeated = Repeat(d, b_repeated)
    a, b = a_repeated, b_repeated
    
    reduction_dims = [a_dims_by_name[name] for name in common_dims if name not in rhs_dim_names]

    match reduction:
        case "sum":
            binary_op = "*"
        case "max":
            binary_op = "max"

    c = BinaryOp(
        op=binary_op, 
        lhs=a,
        rhs=b,
    )
    for d in reduction_dims:
        c = Reduce(reduction, d, c)

    return c

def _construct_unary_reduction_expr(
    a : tuple[DimType, ExprType],
    rhs : DimType,
    reduction : ReduceOpType = "sum",
) -> ExprType:
    dims_a, expr_a = a
    a_dims_by_name = { dim_name(d) : d for d in dims_a }
    rhs_dims_by_name = { dim_name(d) : d for d in rhs }

    repeat_dims = [d for n, d in rhs_dims_by_name.items() if n not in a_dims_by_name]
    reduction_dims = [d for n, d in a_dims_by_name.items() if n not in rhs_dims_by_name]

    result = expr_a
    for d in repeat_dims:
        result = Repeat(d, result)
    for d in reduction_dims:
        result = Reduce(reduction, d, result)
    return result


def _normalize_dts_for_binary_op(lhs : DimType, rhs : DimType) -> DimType:
    if lhs == tuple():
        return rhs
    if rhs == tuple():
        return lhs
    if lhs != rhs:
        raise ValueError("Invalid spec: mismatching DimTypes for binary operation")
    return lhs

def _parse_number(lex : LexState) -> tuple[DimType, ExprType]:
    lex.consume_whitespace()
    i = 0
    while i < len(lex.spec) and (lex.spec[i].isdigit() or lex.spec[i] == '.'):
        i += 1
    
    constant = float(lex.spec[:i])
    lex.spec = lex.spec[i:]
    return tuple(), Constant(constant)

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

def _parse_tensor(lex : LexState) -> tuple[DimType, ExprType]:
    dims = []
    while (nxt := lex.peek()) is not None and nxt.isalpha():
        d = _parse_dim(lex)
        dims.append(d)
    return tuple(dims), Tensor(tuple(dims))

def _parse_contraction(lex : LexState, reduction : ReduceOpType = "sum") -> tuple[DimType, ExprType]:
    lhs_dt, lhs_et = _parse_spec(lex)
    if lex.maybe_consume(','):
        rhs_dt, rhs_et = _parse_spec(lex)
        lex.expect('->')
        result_dt, _ = _parse_tensor(lex)
        return result_dt, _construct_binary_reduction_expr(
            (lhs_dt, lhs_et),
            (rhs_dt, rhs_et),
            result_dt,
            reduction=reduction,
        )
    elif lex.maybe_consume('->'):
        result_dt, _ = _parse_tensor(lex)
        return result_dt, _construct_unary_reduction_expr(
            (lhs_dt, lhs_et),
            result_dt,
            reduction=reduction,
        )
    else:
        raise ValueError("Expected , for binary reduction or -> for unary reduction")

def _parse_paren_expr(lex : LexState) -> tuple[DimType, ExprType]:
    lhs_dt, lhs_et = _parse_spec(lex)
    if lex.maybe_consume(','):
        rhs_dt, rhs_et = _parse_spec(lex)
        lex.expect('->')
        result_dt, _ = _parse_tensor(lex)
        return result_dt, _construct_binary_reduction_expr(
            (lhs_dt, lhs_et),
            (rhs_dt, rhs_et),
            result_dt,
        )
    elif lex.maybe_consume('->'):
        result_dt, _ = _parse_tensor(lex)
        return result_dt, _construct_unary_reduction_expr(
            (lhs_dt, lhs_et),
            result_dt,
        )
    else:
        return lhs_dt, lhs_et
    
def _parse_primary(lex : LexState) -> tuple[DimType, ExprType]:
    if lex.peek() == '(':
        lex.consume()
        dt, et = _parse_paren_expr(lex)
        lex.expect(')')
        return dt, et
    elif (nxt := lex.peek()) is not None:
        if nxt.isalpha():
            return _parse_tensor(lex)
        elif nxt.isdigit():
            return _parse_number(lex)
    raise ValueError("Parenthesized expression, tensor, or number expected")

def _parse_factor(lex : LexState) -> tuple[DimType, ExprType]:
    if reduction := lex.maybe_consume("sum", "max"):
        if lex.maybe_consume('['):
            dim = _parse_dim(lex)
            lex.expect(']')
            lex.expect('(')
            dt, et = _parse_paren_expr(lex)
            lex.expect(')')
            reduce_dt = tuple(d for d in dt if dim_full_dim(d) != dim_full_dim(dim))
            reduce_et = Reduce(
                reduction,
                dim,
                et,
            )
            return reduce_dt, reduce_et
        else:
            lex.expect('(')
            dt, et = _parse_contraction(lex, reduction=reduction)
            lex.expect(')')
            return dt, et
    elif unary_op := lex.maybe_consume("exp", "sin", "cos", "sqrt", "softmax"):
        if lex.maybe_consume('['):
            if unary_op != "softmax":
                raise ValueError("Dimension annotation only makes sense for softmax")

            dim_annotation = _parse_dim(lex)
            lex.expect(']')

        lex.expect('(')
        dt, et = _parse_paren_expr(lex)
        lex.expect(')')
        if unary_op == "softmax":
            exp = UnaryOp(op="exp", child=et)
            sum_exp = Repeat(
                dim=dim_annotation,
                child=Reduce(
                    op="sum",
                    dim=dim_annotation,
                    child=exp,
                ),
            )
            return dt, BinaryOp(
                op="/",
                lhs=exp,
                rhs=sum_exp,
            )

        return dt, UnaryOp(
            op=unary_op,
            child=et,
        )
    else:
        return _parse_primary(lex)

def _parse_term(lex : LexState) -> tuple[DimType, ExprType]:
    result_dt, result_et = _parse_factor(lex)
    while op := lex.maybe_consume("*", "/"):
        rhs_dt, rhs_et = _parse_factor(lex)
        result_dt = _normalize_dts_for_binary_op(result_dt, rhs_dt)
        result_et = BinaryOp(
            op=op,
            lhs=result_et,
            rhs=rhs_et,
        )
    return result_dt, result_et

def _parse_expr(lex : LexState) -> tuple[DimType, ExprType]:
    result_dt, result_et = _parse_term(lex)
    while not lex.startswith("->") and (op := lex.maybe_consume('+', '-')):
        rhs_dt, rhs_et = _parse_term(lex)
        result_dt = _normalize_dts_for_binary_op(result_dt, rhs_dt)
        result_et = BinaryOp(
            op=op,
            lhs=result_et,
            rhs=rhs_et,
        )
    
    return result_dt, result_et    

def _parse_spec(lex : LexState) -> tuple[DimType, ExprType]:
    return _parse_expr(lex)

def parse_spec_into_type(spec : str) -> tuple[DimType, ExprType]:
    """
    Grammar:
    Spec      -> Expr
    Expr      -> Term Expr'
    Expr'     -> '+' Term Expr' | '-' Term Expr' | ε

    Term      -> Factor Term'
    Term'     -> '*' Factor Term' | '/' Factor Term' | ε

    Factor    -> REDUCE_OP '(' Contraction ')' | REDUCE_OP DimAnnot '(' ParenExpr ')' | UNARY_FN DimAnnot? '(' ParenExpr ')' | Primary
    DimAnnot  -> '[' DIM ']'
    Primary   -> '(' ParenExpr ')' | Tensor | Number

    ParenExpr -> Spec ParenExpr'
    ParenExpr'-> ',' Spec '->' Tensor | '->' Tensor | ε

    Contraction -> Spec Contraction'
    Contraction'-> ',' Spec '->' Tensor | '->' Tensor

    Tensor    -> DIM Tensor'
    Tensor'   -> DIM Tensor' | ε

    DIM       -> [A-Z][a-z0-9]*
    Number    -> [0-9]+ ('.' [0-9]+)?
    UNARY_FN  -> 'exp' | 'sin' | 'cos' | 'sqrt' | 'softmax'
    REDUCE_OP -> 'sum' | 'max'
    """
    lex = LexState(spec)
    return _parse_spec(lex)

def remap_dims_by_name(dims_by_name : dict[str, Dim], expr : ExprType) -> ExprType:
    match expr:
        case Constant(_):
            return expr
        case Tensor(_):
            return expr
        case UnaryOp(op, child):
            new_child = remap_dims_by_name(dims_by_name, child)
            return UnaryOp(
                op,
                new_child,
            )
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
        case Reduce(op, d, child):
            name = dim_name(d)
            new_dim = dims_by_name[name] if name in dims_by_name else d
            new_child = remap_dims_by_name(dims_by_name, child)
            return Reduce(op, new_dim, new_child)

def map_expr_to_dim_type(dim_type : tuple[Dim, ...], expr : ExprType) -> ExprType:
    dims_by_name = { dim_name(d) : d for d in dim_type }
    return remap_dims_by_name(dims_by_name, expr)

def expr_types_are_equivalent(dim_type : tuple[Dim, ...], expected : ExprType, actual : ExprType) -> bool:
    expected = map_expr_to_dim_type(dim_type, expected)

    egraph = Egraph()
    expected_id = egraph.insert_expression(expected)
    actual_id = egraph.insert_expression(actual)
    return egraph.incrementally_check_equivalence(expected_id, actual_id)

def expr_simplifies(expr : Typed, spec : str, niters : int = 10) -> bool:
    spec_dt, spec_et = parse_spec_into_type(spec)
    egraph = Egraph()
    expected_id = egraph.insert_expression(expr.expr_type)
    actual_id = egraph.insert_expression(spec_et)
    return egraph.incrementally_check_equivalence(expected_id, actual_id, max_iters=niters)

class TypedResult:
    def __init__(self, spec : str):
        self.expected_dim_type, self.expected_expr_type = parse_spec_into_type(spec)
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

def reset_typed_numpy():
    g_dim_registry.clear()
