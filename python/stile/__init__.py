from .verification import verify_exprs_equivalent
from .specification import parse_spec_into_type
from .type import Type, FullDim, g_dim_registry

def dim(name : str, size : int) -> FullDim:
    return FullDim(name, size)

def expr_simplifies(
    expr : Type,
    spec : str,
) -> bool:
    spec_type = parse_spec_into_type(spec)
    return verify_exprs_equivalent(expr.type.et, spec_type.et)

def reset_stile():
    g_dim_registry.clear()
