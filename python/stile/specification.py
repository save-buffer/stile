from .type import *
from .indexing import (
    AffineExpr, LoopVariable, Domain, to_affine,
    domain as _indexing_domain,
    le, lt, ge, gt, eq, free_vars,
)

def construct_softmax(child : ExprType, dim : Dim):
    if False:
        mx = Repeat(
            dim=dim_full_dim(dim),
            child=Reduce(
                op="max",
                dim=dim,
                child=child,
            ),
        )
        centered = BinaryOp(
            op="-",
            lhs=child,
            rhs=mx,
        )
    else:
        centered = child
    exp = UnaryOp(op="exp", child=centered)
    sum_exp = Repeat(
        dim=dim_full_dim(dim),
        child=Reduce(
            op="sum",
            dim=dim,
            child=exp,
        ),
    )
    return BinaryOp(
        op="/",
        lhs=exp,
        rhs=sum_exp,
    )

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

def _construct_binary_reduction_expr(
    a_type : tuple[ShapeType, ExprType],
    b_type : tuple[ShapeType, ExprType],
    rhs : ShapeType,
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
            a_repeated = Repeat(dim_full_dim(d), a_repeated)

    for name, d in a_dims_by_name.items():
        if name not in common_dims:
            b_repeated = Repeat(dim_full_dim(d), b_repeated)
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
    a : tuple[ShapeType, ExprType],
    rhs : ShapeType,
    reduction : ReduceOpType = "sum",
) -> ExprType:
    dims_a, expr_a = a
    a_dims_by_name = { dim_name(d) : d for d in dims_a }
    rhs_dims_by_name = { dim_name(d) : d for d in rhs }

    repeat_dims = [d for n, d in rhs_dims_by_name.items() if n not in a_dims_by_name]
    reduction_dims = [d for n, d in a_dims_by_name.items() if n not in rhs_dims_by_name]

    result = expr_a
    for d in repeat_dims:
        result = Repeat(dim_full_dim(d), result)
    for d in reduction_dims:
        result = Reduce(reduction, d, result)
    return result


def _normalize_dts_for_binary_op(lhs : ShapeType, rhs : ShapeType) -> ShapeType:
    if lhs == tuple():
        return rhs
    if rhs == tuple():
        return lhs
    if lhs != rhs:
        raise ValueError("Invalid spec: mismatching ShapeTypes for binary operation")
    return lhs

def _parse_number(lex : LexState) -> tuple[ShapeType, ExprType]:
    lex.consume_whitespace()
    i = 0
    while i < len(lex.spec) and (lex.spec[i].isdigit() or lex.spec[i] == '.'):
        i += 1
    
    constant = float(lex.spec[:i])
    lex.spec = lex.spec[i:]
    return tuple(), Constant(constant)

def _parse_integer(lex : LexState) -> int:
    lex.consume_whitespace()
    i = 0
    while i < len(lex.spec) and lex.spec[i].isdigit():
        i += 1
    result = int(lex.spec[:i])
    lex.spec = lex.spec[i:]
    return result

def _parse_dim_name(lex : LexState) -> FullDim:
    lex.consume_whitespace()
    i = 0
    while i < len(lex.spec) and lex.spec[i].isalpha():
        i += 1

    dim_name = lex.spec[:i]
    lex.spec = lex.spec[i:]
    if dim_name not in g_dim_registry:
        raise ValueError(f"Parsed dim {dim_name} is not a known dimension!")
    return g_dim_registry[dim_name]

def _parse_dim(lex : LexState) -> Dim:
    dim = _parse_dim_name(lex)
    if lex.maybe_consume('['):
        slice_start = _parse_integer(lex)
        lex.expect(':')
        slice_end = _parse_integer(lex)
        lex.expect(']')
        dim = Sliced(
            dim,
            slice_start,
            slice_end,
        )
    return dim

def _parse_tensor(lex : LexState) -> tuple[ShapeType, ExprType]:
    dims = []
    # Stop when the upcoming identifier isn't a registered dim — otherwise
    # trailing keywords (`where`, …) would be swallowed as dim names.
    while (nxt := lex.peek()) is not None and nxt.isalpha():
        i = 0
        while i < len(lex.spec) and (lex.spec[i].isalpha() or lex.spec[i].isdigit()):
            i += 1
        word = lex.spec[:i]
        if word not in g_dim_registry:
            break
        d = _parse_dim(lex)
        dims.append(d)
    full_dims = tuple(dim_full_dim(d) for d in dims)
    return tuple(dims), Tensor(full_dims)

def _parse_contraction(lex : LexState, reduction : ReduceOpType = "sum") -> tuple[ShapeType, ExprType]:
    lhs_st, lhs_et = _parse_spec(lex)
    if lex.maybe_consume(','):
        rhs_st, rhs_et = _parse_spec(lex)
        lex.expect('->')
        result_st, _ = _parse_tensor(lex)
        return result_st, _construct_binary_reduction_expr(
            (lhs_st, lhs_et),
            (rhs_st, rhs_et),
            result_st,
            reduction=reduction,
        )
    elif lex.maybe_consume('->'):
        result_st, _ = _parse_tensor(lex)
        return result_st, _construct_unary_reduction_expr(
            (lhs_st, lhs_et),
            result_st,
            reduction=reduction,
        )
    else:
        raise ValueError("Expected , for binary reduction or -> for unary reduction")

def _parse_paren_expr(lex : LexState) -> tuple[ShapeType, ExprType]:
    lhs_st, lhs_et = _parse_spec(lex)
    if lex.maybe_consume(','):
        rhs_st, rhs_et = _parse_spec(lex)
        lex.expect('->')
        result_st, _ = _parse_tensor(lex)
        return result_st, _construct_binary_reduction_expr(
            (lhs_st, lhs_et),
            (rhs_st, rhs_et),
            result_st,
        )
    elif lex.maybe_consume('->'):
        result_st, _ = _parse_tensor(lex)
        return result_st, _construct_unary_reduction_expr(
            (lhs_st, lhs_et),
            result_st,
        )
    else:
        return lhs_st, lhs_et
    
def _parse_primary(lex : LexState) -> tuple[ShapeType, ExprType]:
    if lex.peek() == '(':
        lex.consume()
        st, et = _parse_paren_expr(lex)
        lex.expect(')')
        return st, et
    elif (nxt := lex.peek()) is not None:
        if nxt.isalpha():
            return _parse_tensor(lex)
        elif nxt.isdigit():
            return _parse_number(lex)
    raise ValueError("Parenthesized expression, tensor, or number expected")

def _parse_factor(lex : LexState) -> tuple[ShapeType, ExprType]:
    if reduction := lex.maybe_consume("sum", "max"):
        if lex.maybe_consume('['):
            dim = _parse_dim(lex)
            lex.expect(']')
            lex.expect('(')
            st, et = _parse_paren_expr(lex)
            lex.expect(')')
            reduce_st = tuple(d for d in st if dim_full_dim(d) != dim_full_dim(dim))
            reduce_dim = [d for d in st if dim_full_dim(d) == dim_full_dim(dim)][0]
            reduce_et = Reduce(
                reduction, # ty: ignore
                reduce_dim,
                et,
            )
            return reduce_st, reduce_et
        else:
            lex.expect('(')
            st, et = _parse_contraction(lex, reduction=reduction) # ty: ignore
            lex.expect(')')
            return st, et
    elif unary_op := lex.maybe_consume("exp", "sin", "cos", "sqrt", "softmax"):
        if lex.maybe_consume('['):
            if unary_op != "softmax":
                raise ValueError("Dimension annotation only makes sense for softmax")

            dim_annotation = _parse_dim(lex)
            lex.expect(']')

        lex.expect('(')
        st, et = _parse_paren_expr(lex)
        lex.expect(')')
        if unary_op == "softmax":
            return st, construct_softmax(et, dim_annotation)

        return st, UnaryOp(
            op=unary_op, # ty: ignore
            child=et,
        )
    else:
        return _parse_primary(lex)

def _parse_term(lex : LexState) -> tuple[ShapeType, ExprType]:
    result_st, result_et = _parse_factor(lex)
    while op := lex.maybe_consume("*", "/"):
        rhs_st, rhs_et = _parse_factor(lex)
        result_st = _normalize_dts_for_binary_op(result_st, rhs_st)
        result_et = BinaryOp(
            op=op, # ty: ignore
            lhs=result_et,
            rhs=rhs_et,
        )
    return result_st, result_et

def _parse_expr(lex : LexState) -> tuple[ShapeType, ExprType]:
    result_st, result_et = _parse_term(lex)
    while not lex.startswith("->") and (op := lex.maybe_consume('+', '-')):
        rhs_st, rhs_et = _parse_term(lex)
        result_st = _normalize_dts_for_binary_op(result_st, rhs_st)
        result_et = BinaryOp(
            op=op, # ty: ignore
            lhs=result_et,
            rhs=rhs_et,
        )
    
    return result_st, result_et

def _parse_affine_term(lex : LexState) -> AffineExpr:
    lex.consume_whitespace()
    nxt = lex.peek()
    if nxt is None:
        raise ValueError("Expected affine term in `where`-predicate")
    if nxt.isdigit():
        coef = _parse_integer(lex)
        if lex.maybe_consume('*'):
            d = _parse_dim_name(lex)
            return to_affine(LoopVariable(d.name)) * coef
        return to_affine(coef)
    if nxt.isalpha():
        d = _parse_dim_name(lex)
        return to_affine(LoopVariable(d.name))
    raise ValueError(f"Expected integer or dim name in `where`-predicate, got {nxt!r}")


def _parse_affine(lex : LexState) -> AffineExpr:
    lex.consume_whitespace()
    sign = 1
    if lex.spec.startswith('-') and not lex.spec.startswith('->'):
        lex.consume()
        sign = -1
    elif lex.spec.startswith('+'):
        lex.consume()
    result = _parse_affine_term(lex)
    if sign == -1:
        result = -result
    while True:
        lex.consume_whitespace()
        if lex.spec.startswith('->'):
            break
        if lex.maybe_consume('+'):
            result = result + _parse_affine_term(lex)
        elif lex.maybe_consume('-'):
            result = result - _parse_affine_term(lex)
        else:
            break
    return result


def _parse_predicate(lex : LexState) -> Domain:
    lhs = _parse_affine(lex)
    relop = lex.maybe_consume("<=", ">=", "==", "<", ">")
    if relop is None:
        raise ValueError("Expected comparison operator (<=, <, >=, >, ==) in `where`-predicate")
    rhs = _parse_affine(lex)
    variables = free_vars(lhs) | free_vars(rhs)
    match relop:
        case "<=":
            constraints = [le(lhs, rhs)]
        case ">=":
            constraints = [ge(lhs, rhs)]
        case "<":
            constraints = [lt(lhs, rhs)]
        case ">":
            constraints = [gt(lhs, rhs)]
        case "==":
            c1, c2 = eq(lhs, rhs)
            constraints = [c1, c2]
    return _indexing_domain(variables, constraints)


def _apply_where_mask(
    result_st : ShapeType,
    result_et : ExprType,
    pred_domain : Domain,
) -> ExprType:
    dim_names_in_shape = {dim_name(d) for d in result_st}
    for v in pred_domain.variables:
        if v.name not in dim_names_in_shape:
            raise ValueError(
                f"`where`-clause references dim {v.name!r} which is not "
                f"present in the expression's shape {sorted(dim_names_in_shape)}"
            )
    mask_dims = tuple(dim_full_dim(d) for d in result_st)
    mask = Tensor(
        dims=mask_dims,
        tag=TagCond(
            domain=pred_domain,
            if_true=Constant(1.0),
            if_false=Constant(0.0),
        ),
    )
    return BinaryOp(op="*", lhs=result_et, rhs=mask)


def _parse_spec(lex : LexState) -> tuple[ShapeType, ExprType]:
    result_st, result_et = _parse_expr(lex)
    while lex.maybe_consume("where"):
        pred_domain = _parse_predicate(lex)
        result_et = _apply_where_mask(result_st, result_et, pred_domain)
    return result_st, result_et

def parse_spec_into_type(spec : str) -> Type:
    """
    Grammar:
    Spec      -> Expr ('where' Predicate)*
    Predicate -> Affine RELOP Affine
    Affine    -> [+-]? AffineTerm (('+'|'-') AffineTerm)*
    AffineTerm-> Integer | Integer '*' DimName | DimName
    RELOP     -> '<=' | '<' | '>=' | '>' | '=='
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

    DIM       -> DimName | DimName '[' Integer ']'
    DimName   -> [A-Z][a-z0-9]*
    Number    -> [0-9]+ ('.' [0-9]+)?
    UNARY_FN  -> 'exp' | 'sin' | 'cos' | 'sqrt' | 'softmax'
    REDUCE_OP -> 'sum' | 'max'
    """
    lex = LexState(spec)
    st, et = _parse_spec(lex)
    return Type(st, et)
