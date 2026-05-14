from .type import *
from .type import _g_tensor_counter
from .indexing import (
    AffineExpr, LoopVariable, Domain, to_affine,
    domain as _indexing_domain,
    le, lt, ge, gt, eq, free_vars, and_domains,
)


# Free loop variables visible to the affine-predicate parser. Set by
# `parse_spec_into_type(..., loop_vars=...)` so a spec like
# `sum[N where N < k](X)` can use `k` as a bound LoopVariable that
# isn't in `g_dim_registry`. Restored after the parse.
_g_extra_loop_vars : set[str] = set()

def construct_softmax(
    child : ExprType,
    dim : Dim,
    pred_domain : "Domain | None" = None,
    body_st : "ShapeType | None" = None,
):
    """
    `exp(X) / Σ_d exp(X)`. When `pred_domain` is given (i.e. `softmax[d
    where P]`), the mask is applied **multiplicatively to `exp(X)`**, not
    additively to `X`. The two are algebraically equivalent
    (`exp(X+Cond(P,0,-inf)) = exp(X)·Cond(P,1,0)`) but the multiplicative
    form folds cleanly into both numerator and denominator reduce
    domains, so `softmax[d where P](X)·V → q dhead` canonicalizes to the
    same form a hand-written `Σ exp(X)·V where P / Σ exp(X) where P`
    would produce — and that's what a tiled flash-attention kernel ends
    up computing after rescaling cancels.
    """
    exp = UnaryOp(op="exp", child=child)
    masked_exp : ExprType = exp
    if pred_domain is not None and body_st is not None:
        for v in pred_domain.variables:
            if (
                v.name not in {dim_name(d) for d in body_st}
                and v.name not in _g_extra_loop_vars
            ):
                raise ValueError(
                    f"softmax[{dim_name(dim)} where ...] predicate "
                    f"references dim {v.name!r} not in body shape "
                    f"{sorted({dim_name(d) for d in body_st})}"
                )
        mask = Tensor(
            dims=tuple(dim_full_dim(d) for d in body_st),
            tag=TagCond(
                domain=pred_domain,
                if_true=Constant(1.0),
                if_false=Constant(0.0),
            ),
            name="_mask",
        )
        masked_exp = BinaryOp(op="*", lhs=exp, rhs=mask)
    sum_exp = Repeat(
        dim=dim_full_dim(dim),
        child=Reduce(
            op="sum",
            dim=dim,
            child=masked_exp,
        ),
    )
    return BinaryOp(
        op="/",
        lhs=masked_exp,
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

    def maybe_consume_keyword(self, *args) -> str | None:
        """Like `maybe_consume` but requires a non-identifier boundary
        after the match. Use for keywords (`exp`, `sum`, `gather`, …)
        so an identifier like `expert_id` doesn't get partially
        consumed as `exp` + the rest left dangling."""
        self.consume_whitespace()
        for a in args:
            if self.spec.startswith(a):
                end = len(a)
                if end >= len(self.spec) or not (
                    self.spec[end].isalpha()
                    or self.spec[end].isdigit()
                    or self.spec[end] == "_"
                ):
                    self.spec = self.spec[end:]
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
    if lhs == rhs:
        return lhs
    lhs_names = {dim_name(d) for d in lhs}
    rhs_names = {dim_name(d) for d in rhs}
    if lhs_names == rhs_names:
        # Same dims, possibly different order — pick `lhs`'s order.
        return lhs
    # Generalized broadcast: output gets the union of dim names. Take
    # `lhs`'s dims first (preserving order), then any rhs-only dims.
    lhs_seen : set[str] = set()
    out : list[Dim] = []
    for d in lhs:
        out.append(d)
        lhs_seen.add(dim_name(d))
    for d in rhs:
        if dim_name(d) not in lhs_seen:
            out.append(d)
    return tuple(out)


def _broadcast_to(et : ExprType, src_st : ShapeType, dst_st : ShapeType) -> ExprType:
    """Wrap `et` in `Repeat`s so that it matches `dst_st`'s dims. `src_st`
    must have a subset of `dst_st`'s dim names."""
    src_names = {dim_name(d) for d in src_st}
    for d in dst_st:
        if dim_name(d) not in src_names:
            et = Repeat(dim=dim_full_dim(d), child=et)
    return et


def _binary_op_with_broadcast(
    op : str,
    lhs_st : ShapeType, lhs_et : ExprType,
    rhs_st : ShapeType, rhs_et : ExprType,
) -> tuple[ShapeType, ExprType]:
    out_st = _normalize_dts_for_binary_op(lhs_st, rhs_st)
    if out_st != lhs_st and lhs_st != tuple():
        lhs_et = _broadcast_to(lhs_et, lhs_st, out_st)
    if out_st != rhs_st and rhs_st != tuple():
        rhs_et = _broadcast_to(rhs_et, rhs_st, out_st)
    return out_st, BinaryOp(op=op, lhs=lhs_et, rhs=rhs_et)  # ty: ignore

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

def _is_ident_start(c : str) -> bool:
    return c.isalpha() or c == "_"


def _is_ident_cont(c : str) -> bool:
    return c.isalpha() or c.isdigit() or c == "_"


def _parse_dim_name(lex : LexState) -> FullDim:
    lex.consume_whitespace()
    if not lex.spec or not _is_ident_start(lex.spec[0]):
        raise ValueError(f"Expected dim name at {lex.spec[:20]!r}")
    i = 1
    while i < len(lex.spec) and _is_ident_cont(lex.spec[i]):
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

def _parse_shape(lex : LexState) -> ShapeType:
    """
    Parse a sequence of dim references into a `ShapeType` *without*
    constructing a Tensor — used for `-> ResultShape` slots where the
    parser only needs the shape, not a tensor identity.
    """
    dims = []
    # Stop when the upcoming identifier isn't a registered dim — otherwise
    # trailing keywords (`where`, …) would be swallowed as dim names.
    while (nxt := lex.peek()) is not None and _is_ident_start(nxt):
        i = 1
        while i < len(lex.spec) and _is_ident_cont(lex.spec[i]):
            i += 1
        word = lex.spec[:i]
        if word not in g_dim_registry:
            break
        d = _parse_dim(lex)
        dims.append(d)
    return tuple(dims)


def _maybe_parse_label(lex : LexState) -> str | None:
    """
    Optional `label:` prefix on a tensor reference. Two `label:dims`
    occurrences with the same label refer to the same tensor leaf. An
    unlabeled tensor gets an auto-generated `_tensor_<n>` name and is
    distinct from every other unlabeled tensor in the spec.
    """
    lex.consume_whitespace()
    i = 0
    while i < len(lex.spec) and (
        lex.spec[i].isalpha() or lex.spec[i].isdigit() or lex.spec[i] == '_'
    ):
        i += 1
    if i == 0:
        return None
    if i >= len(lex.spec) or lex.spec[i] != ':':
        return None
    label = lex.spec[:i]
    lex.spec = lex.spec[i+1:]
    return label


def _parse_tensor(lex : LexState) -> tuple[ShapeType, ExprType]:
    name = _maybe_parse_label(lex)
    dims = _parse_shape(lex)
    full_dims = tuple(dim_full_dim(d) for d in dims)
    return dims, Tensor(full_dims, name=name)

def _parse_contraction(lex : LexState, reduction : ReduceOpType = "sum") -> tuple[ShapeType, ExprType]:
    lhs_st, lhs_et = _parse_spec(lex)
    if lex.maybe_consume(','):
        rhs_st, rhs_et = _parse_spec(lex)
        lex.expect('->')
        result_st = _parse_shape(lex)
        return result_st, _construct_binary_reduction_expr(
            (lhs_st, lhs_et),
            (rhs_st, rhs_et),
            result_st,
            reduction=reduction,
        )
    elif lex.maybe_consume('->'):
        result_st = _parse_shape(lex)
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
        result_st = _parse_shape(lex)
        return result_st, _construct_binary_reduction_expr(
            (lhs_st, lhs_et),
            (rhs_st, rhs_et),
            result_st,
        )
    elif lex.maybe_consume('->'):
        result_st = _parse_shape(lex)
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
    if reduction := lex.maybe_consume_keyword("sum", "max"):
        if lex.maybe_consume('['):
            dim = _parse_dim(lex)
            pred_domain = None
            if lex.maybe_consume_keyword("where"):
                pred_domain = _parse_predicate(lex)
            lex.expect(']')
            lex.expect('(')
            st, et = _parse_paren_expr(lex)
            lex.expect(')')
            reduce_dim = [d for d in st if dim_full_dim(d) == dim_full_dim(dim)][0]
            if pred_domain is not None:
                # When the predicate is a "pure iteration restriction" —
                # a per-direction affine bound on the reduce dim whose
                # other free variables don't appear in the body's shape
                # — lower it as a `Sliced` reduce instead of a
                # multiplicative mask. This skips the mask-fold round
                # trip and produces a cleaner reduce domain that
                # downstream tile-merge composes through. Mask form
                # survives for cases where the predicate references
                # body dims (causal flash).
                slice_bounds = _try_lower_predicate_to_slice(
                    pred_domain, dim, st,
                )
                if slice_bounds is not None:
                    lo, hi = slice_bounds
                    reduce_dim = Sliced(dim_full_dim(reduce_dim), lo, hi)
                else:
                    et = _apply_iteration_restriction(
                        st, et, pred_domain, reduction,
                    )
            reduce_st = tuple(d for d in st if dim_full_dim(d) != dim_full_dim(dim))
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
    elif lex.maybe_consume_keyword("gather"):
        # gather[dim_in_source](source_expr, idx_expr)
        # The result's shape replaces `dim_in_source` (which must appear
        # in source's shape) with `idx`'s sole dim.
        lex.expect('[')
        dim_in_source = _parse_dim(lex)
        lex.expect(']')
        lex.expect('(')
        source_st, source_et = _parse_spec(lex)
        lex.expect(',')
        idx_st, idx_et = _parse_spec(lex)
        lex.expect(')')
        if len(idx_st) != 1:
            raise ValueError(
                f"gather index must be 1-d; got shape={idx_st}"
            )
        if not any(dim_name(d) == dim_name(dim_in_source) for d in source_st):
            raise ValueError(
                f"gather dim {dim_name(dim_in_source)!r} not in source "
                f"shape {[dim_name(d) for d in source_st]}"
            )
        out_st = tuple(
            idx_st[0] if dim_name(d) == dim_name(dim_in_source) else d
            for d in source_st
        )
        return out_st, Gather(
            source=source_et,
            dim_in_source=dim_full_dim(dim_in_source),
            idx=idx_et,
        )
    elif lex.maybe_consume_keyword("scatter"):
        # scatter[dim_in_dest](source_expr, idx_expr)
        # Dual of gather: source has idx's dim in its shape; the result
        # replaces that dim with `dim_in_dest`. Positions of the output
        # not addressed by `idx` are zero.
        lex.expect('[')
        dim_in_dest = _parse_dim(lex)
        lex.expect(']')
        lex.expect('(')
        source_st, source_et = _parse_spec(lex)
        lex.expect(',')
        idx_st, idx_et = _parse_spec(lex)
        lex.expect(')')
        if len(idx_st) != 1:
            raise ValueError(
                f"scatter index must be 1-d; got shape={idx_st}"
            )
        idx_dim = idx_st[0]
        if not any(dim_name(d) == dim_name(idx_dim) for d in source_st):
            raise ValueError(
                f"scatter source must have idx's dim "
                f"{dim_name(idx_dim)!r} in shape "
                f"{[dim_name(d) for d in source_st]}"
            )
        out_st = tuple(
            dim_in_dest if dim_name(d) == dim_name(idx_dim) else d
            for d in source_st
        )
        return out_st, Scatter(
            source=source_et,
            dim_in_dest=dim_full_dim(dim_in_dest),
            idx=idx_et,
        )
    elif unary_op := lex.maybe_consume_keyword("exp", "sin", "cos", "sqrt", "softmax"):
        pred_domain = None
        if lex.maybe_consume('['):
            if unary_op != "softmax":
                raise ValueError("Dimension annotation only makes sense for softmax")

            dim_annotation = _parse_dim(lex)
            if lex.maybe_consume_keyword("where"):
                pred_domain = _parse_predicate(lex)
            lex.expect(']')

        lex.expect('(')
        st, et = _parse_paren_expr(lex)
        lex.expect(')')
        if unary_op == "softmax":
            return st, construct_softmax(et, dim_annotation, pred_domain, st)

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
        result_st, result_et = _binary_op_with_broadcast(
            op, result_st, result_et, rhs_st, rhs_et,
        )
    return result_st, result_et

def _parse_expr(lex : LexState) -> tuple[ShapeType, ExprType]:
    result_st, result_et = _parse_term(lex)
    while not lex.startswith("->") and (op := lex.maybe_consume('+', '-')):
        rhs_st, rhs_et = _parse_term(lex)
        result_st, result_et = _binary_op_with_broadcast(
            op, result_st, result_et, rhs_st, rhs_et,
        )
    return result_st, result_et

def _parse_affine_identifier(lex : LexState) -> LoopVariable:
    """
    Read an identifier and return it as a `LoopVariable`. Looks up
    `_g_extra_loop_vars` first (free invariant-style loop vars like
    `k`), falling back to the dim registry — that way `nctx <= qctx`
    still resolves dim names while a fori_loop invariant can mention
    a non-dim `k`.
    """
    lex.consume_whitespace()
    if not lex.spec or not _is_ident_start(lex.spec[0]):
        d = _parse_dim_name(lex)
        return LoopVariable(d.name)
    i = 1
    while i < len(lex.spec) and _is_ident_cont(lex.spec[i]):
        i += 1
    word = lex.spec[:i]
    if word in _g_extra_loop_vars:
        lex.spec = lex.spec[i:]
        return LoopVariable(word)
    d = _parse_dim_name(lex)
    return LoopVariable(d.name)


def _parse_affine_term(lex : LexState) -> AffineExpr:
    lex.consume_whitespace()
    nxt = lex.peek()
    if nxt is None:
        raise ValueError("Expected affine term in `where`-predicate")
    if nxt.isdigit():
        coef = _parse_integer(lex)
        if lex.maybe_consume('*'):
            v = _parse_affine_identifier(lex)
            return to_affine(v) * coef
        return to_affine(coef)
    if nxt.isalpha():
        v = _parse_affine_identifier(lex)
        return to_affine(v)
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


def _parse_atomic_predicate(lex : LexState) -> Domain:
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


def _parse_predicate(lex : LexState) -> Domain:
    """
    A `where`-clause predicate, optionally a conjunction joined by `&&`
    (or `and`). The result is a single-disjunct `Domain` whose conjunct
    is the AND of every atomic comparison.
    """
    domain = _parse_atomic_predicate(lex)
    while (lex.maybe_consume("&&") or lex.maybe_consume_keyword("and")):
        rhs = _parse_atomic_predicate(lex)
        domain = and_domains(domain, rhs)
    return domain


def _apply_where_mask(
    result_st : ShapeType,
    result_et : ExprType,
    pred_domain : Domain,
) -> ExprType:
    dim_names_in_shape = {dim_name(d) for d in result_st}
    for v in pred_domain.variables:
        if v.name not in dim_names_in_shape and v.name not in _g_extra_loop_vars:
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
        name="_mask",
    )
    return BinaryOp(op="*", lhs=result_et, rhs=mask)


def _try_lower_predicate_to_slice(
    pred_domain : Domain,
    dim : Dim,
    body_st : ShapeType,
) -> tuple[SymbolicIndex, SymbolicIndex] | None:
    """
    If `pred_domain` is a "pure iteration restriction" on `dim`, return
    `(lower, upper)` slice bounds; else None (caller falls back to the
    multiplicative-mask form).

    Pure iteration restriction means:
      - Single-conjunct predicate.
      - Each constraint is a per-direction affine bound on the reduce
        dim (`coeff = ±1` on the reduce var, no other vars with that
        dim, simple lower-or-upper).
      - The bound's residue can reference free variables, but those
        variables must NOT also appear in `body_st`'s dim names. Body
        dims would mean the bound varies with the body's iteration
        domain — a true mask, not a slice.
      - At most one bound per direction. Multiple bounds would require
        a symbolic `min`/`max` to pick the tightest, which we don't
        try to construct.
    """
    reduce_var_name = dim_name(dim)
    reduce_var = LoopVariable(reduce_var_name)
    body_dim_names = {dim_name(d) for d in body_st}
    body_dim_names_excl_reduce = body_dim_names - {reduce_var_name}
    for v in pred_domain.variables:
        if v.name in body_dim_names_excl_reduce:
            return None
    if len(pred_domain.disjuncts) != 1:
        return None
    conj = next(iter(pred_domain.disjuncts))

    lower_bounds : list[SymbolicIndex] = []
    upper_excl_bounds : list[SymbolicIndex] = []
    for c in conj:
        aff = c.expr
        var_coeff = 0
        for v, co in aff.terms:
            if v == reduce_var:
                var_coeff = co
        if var_coeff not in (1, -1):
            return None
        if var_coeff == 1:
            residue = aff - to_affine(reduce_var)
            lower_bounds.append(-residue)
        else:
            residue = aff + to_affine(reduce_var)
            upper_excl_bounds.append(residue + to_affine(1))
    if len(lower_bounds) > 1 or len(upper_excl_bounds) > 1:
        return None

    lower = lower_bounds[0] if lower_bounds else dim_start(dim)
    upper = upper_excl_bounds[0] if upper_excl_bounds else dim_end(dim)
    return (lower, upper)


def _apply_iteration_restriction(
    body_st : ShapeType,
    body_et : ExprType,
    pred_domain : Domain,
    op : str,
) -> ExprType:
    """
    Lower a `[d where P]` dim-annotation to the appropriate mask shape for
    the surrounding op:
      - `sum[d where P]`     → multiplicative mask: body * Cond(P, 1, 0).
      - `max[d where P]`     → bias mask: body + Cond(P, 0, -inf).
      - `softmax[d where P]` → bias mask (applied before macro expansion);
        bias-form `-inf` reaches both num and den so the softmax is
        properly restricted to the predicate.
    """
    dim_names_in_shape = {dim_name(d) for d in body_st}
    for v in pred_domain.variables:
        if v.name not in dim_names_in_shape and v.name not in _g_extra_loop_vars:
            raise ValueError(
                f"`{op}[d where P]` predicate references dim {v.name!r} "
                f"not in the body's shape {sorted(dim_names_in_shape)}"
            )
    mask_dims = tuple(dim_full_dim(d) for d in body_st)
    if op == "sum":
        mask = Tensor(
            dims=mask_dims,
            tag=TagCond(
                domain=pred_domain,
                if_true=Constant(1.0),
                if_false=Constant(0.0),
            ),
            name="_mask",
        )
        return BinaryOp(op="*", lhs=body_et, rhs=mask)
    # max, softmax: bias-form. -inf outside the predicate vanishes through
    # exp (softmax) and is the identity for max.
    bias = Tensor(
        dims=mask_dims,
        tag=TagCond(
            domain=pred_domain,
            if_true=Constant(0.0),
            if_false=Constant(float("-inf")),
        ),
        name="_mask",
    )
    return BinaryOp(op="+", lhs=body_et, rhs=bias)


def _parse_spec(lex : LexState) -> tuple[ShapeType, ExprType]:
    result_st, result_et = _parse_expr(lex)
    while lex.maybe_consume_keyword("where"):
        pred_domain = _parse_predicate(lex)
        result_et = _apply_where_mask(result_st, result_et, pred_domain)
    return result_st, result_et

def parse_spec_into_type(
    spec : str,
    *,
    loop_vars : set[str] | None = None,
) -> Type:
    """
    `loop_vars` lists identifier names that should be treated as free
    `LoopVariable`s during this parse — typically a fori_loop's index
    variable (`{"k"}`). Without this, those names would fall through
    to the dim registry and error.

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
    DimAnnot  -> '[' DIM ('where' Predicate)? ']'
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
    # Save / reset / restore the tensor-naming counter so each spec
    # parses with auto-names starting from `_tensor_1`. The kernel side
    # also starts at 1 after `reset_stile`, so spec-leaf names align
    # with kernel-leaf names by construction order.
    global _g_extra_loop_vars
    saved_counter = _g_tensor_counter[0]
    _g_tensor_counter[0] = 0
    saved_loop_vars = _g_extra_loop_vars
    _g_extra_loop_vars = saved_loop_vars | (loop_vars or set())
    try:
        lex = LexState(spec)
        st, et = _parse_spec(lex)
        return Type(st, et)
    finally:
        _g_tensor_counter[0] = saved_counter
        _g_extra_loop_vars = saved_loop_vars
