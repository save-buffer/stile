from collections import Counter
from dataclasses import dataclass

from .type import *
from ._rust import RustEgraph, RustExpr # ty: ignore

def expr_type_to_rust_expr(expr : ExprType) -> RustExpr:
    match expr:
        case Constant(v):
            return RustExpr.Constant(v)
        case Tensor(dims):
            return RustExpr.Tensor([dim_name(d) for d in dims])
        case UnaryOp(op, child):
            rust_child = expr_type_to_rust_expr(child)
            match op:
                case "exp":
                    return RustExpr.Exp(rust_child)
                case "sin":
                    return RustExpr.Sin(rust_child)
                case "cos":
                    return RustExpr.Cos(rust_child)
                case "sqrt":
                    return RustExpr.Sqrt(rust_child)
            raise ValueError(f"Unknown unary op {op}")
        case BinaryOp(op, lhs, rhs):
            rust_lhs = expr_type_to_rust_expr(lhs)
            rust_rhs = expr_type_to_rust_expr(rhs)
            match op:
                case "+":
                    return RustExpr.Add(rust_lhs, rust_rhs)
                case "-":
                    return RustExpr.Sub(rust_lhs, rust_rhs)
                case "*":
                    return RustExpr.Mul(rust_lhs, rust_rhs)
                case "/":
                    return RustExpr.Div(rust_lhs, rust_rhs)
                case "max":
                    return RustExpr.BinaryMax(rust_lhs, rust_rhs)
            raise ValueError(f"Unknown binary op {op}")
        case Repeat(dim, child):
            d = dim_name(dim)
            rust_child = expr_type_to_rust_expr(child)
            return RustExpr.Repeat(d, rust_child)
        case Reduce(op, dim, child):
            d = dim_name(dim)
            start, end = dim_start(dim), dim_end(dim)
            rust_child = expr_type_to_rust_expr(child)
            match op:
                case "sum":
                    return RustExpr.Sum(d, start, end, rust_child)
                case "max":
                    return RustExpr.Max(d, start, end, rust_child)
            raise ValueError(f"Unknown reduction op {op}")

@dataclass(frozen=True)
class NormalizedTensor:
    dims : frozenset[FullDim]

@dataclass(frozen=True)
class NormalizedExp:
    child : "NormalizedExpr"

@dataclass(frozen=True)
class NormalizedUnaryOp:
    op : UnaryOpType
    child : "NormalizedExpr"

@dataclass(frozen=True)
class NormalizedSum:
    children : frozenset["NormalizedProduct"]

@dataclass(frozen=True)
class NormalizedMax:
    children : frozenset["NormalizedExpr"]

@dataclass(frozen=True)
class NormalizedRepeat:
    dims : frozenset[FullDim]
    child : "NormalizedExpr"

@dataclass(frozen=True)
class NormalizedReduce:
    dim : FullDim
    op : ReduceOpType
    intervals : tuple[tuple[int, int], ...]
    child : "NormalizedExpr"

NormalizedFactor = Tensor | NormalizedExp | NormalizedUnaryOp | NormalizedSum | NormalizedMax | NormalizedRepeat

@dataclass(frozen=True)
class NormalizedProduct:
    const : float = 1.0
    factors : Counter[NormalizedFactor] = field(default_factory=Counter)

@dataclass(frozen=True)
class NormalizedExpr:
    num : NormalizedProduct
    den : NormalizedProduct

    def __post_init__(self):
        assert self.den.const == 1.0

    @staticmethod
    def from(x : NormalizedProduct | NormalizedFactor) -> "NormalizedExpr":
        match x:
            case NormalizedProduct:
                num = x
                den = NormalizedProduct(const=1.0)
                return NormalizedExpr(num, den)
            case NormalizedFactor:
                num = NormalizedProduct(factors=frozenset({x}))
                den = NormalizedProduct(const=1.0)
                return NormalizedExpr(num, den)

def normalize(expr : ExprType) -> NormalizedExpr:
    match expr:
        case Constant(x):
            const = NormalizedProduct(const=x)
            return NormalizedExpr.from(const)
        case Tensor(dims):
            t = NormalizedTensor(dims=frozenset(dims))
            return NormalizedProduct.from(t)
        case UnaryOp(op, child):
            normalized_child = normalize(child)
            unary = (
                NormalizedExp(normalized_child) 
                if op == "exp" 
                else NormalizedUnaryOp(op, normalized_child)
            )
            return NormalizedExpr.from(unary)
        case BinaryOp(op, lhs, rhs):
            nlhs = normalize(lhs)
            nrhs = normalize(rhs)
            lnum, lden = nlhs.num, nlhs.den
            rnum, rden = nrhs.num, nrhs.den
            match op:
                case "+":
                    new_num = NormalizedProduct(
                        const=(lnum.const * rden.const) / (lden.const * rnum.const),
                        factors=lnum.factors + rnum.factors,
                    )

def verify_exprs_equivalent(x : ExprType, y : ExprType) -> bool:
    egg = RustEgraph()
    x_rust = expr_type_to_rust_expr(x)
    y_rust = expr_type_to_rust_expr(y)
    x_id = egg.insert_expression(x_rust)
    y_id = egg.insert_expression(y_rust)
    return egg.incrementally_check_equivalence(x_id, y_id)

def verify_dims_equivalent(x : ShapeType, y : ShapeType) -> bool:
    if len(x) != len(y):
        return False
    for x, y in zip(x, y, strict=True):
        if dim_full_dim(x) != dim_full_dim(y):
            return False
    return True

def verify_types_equivalent(x : Type, y : Type) -> bool:
    dim_types_match = verify_dims_equivalent(x.st, y.st)
    if not dim_types_match:
        return False

    expr_types_match = verify_exprs_equivalent(x.et, y.et)
    return expr_types_match
