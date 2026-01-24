from .type import *

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
        result = Repeat(dim_full_dim(d), result)
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

def _parse_tensor(lex : LexState) -> tuple[DimType, ExprType]:
    dims = []
    while (nxt := lex.peek()) is not None and nxt.isalpha():
        d = _parse_dim(lex)
        dims.append(d)
    full_dims = tuple(dim_full_dim(d) for d in dims)
    return tuple(dims), Tensor(full_dims)

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
            reduce_dim = [d for d in dt if dim_full_dim(d) == dim_full_dim(dim)][0]
            reduce_et = Reduce(
                reduction, # ty: ignore
                reduce_dim,
                et,
            )
            return reduce_dt, reduce_et
        else:
            lex.expect('(')
            dt, et = _parse_contraction(lex, reduction=reduction) # ty: ignore
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
                dim=dim_full_dim(dim_annotation),
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
            op=unary_op, # ty: ignore
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
            op=op, # ty: ignore
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
            op=op, # ty: ignore
            lhs=result_et,
            rhs=rhs_et,
        )
    
    return result_dt, result_et

def _parse_spec(lex : LexState) -> tuple[DimType, ExprType]:
    return _parse_expr(lex)

def parse_spec_into_type(spec : str) -> Type:
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

    DIM       -> DimName | DimName '[' Integer ']'
    DimName   -> [A-Z][a-z0-9]*
    Number    -> [0-9]+ ('.' [0-9]+)?
    UNARY_FN  -> 'exp' | 'sin' | 'cos' | 'sqrt' | 'softmax'
    REDUCE_OP -> 'sum' | 'max'
    """
    lex = LexState(spec)
    dt, et = _parse_spec(lex)
    return Type(dt, et)
