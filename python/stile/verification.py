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
