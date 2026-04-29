import functools
import math
from dataclasses import dataclass, field

from .type import *
from .indexing import Domain, AffineConstraint, interval_domain, ge, lt, and_domains, simplify_domain
from .frozen_counter import FrozenCounter
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
class NormalizedTagCond:
    """
    Branch of a `NormalizedTagTree`: on `domain`, use `if_true`; else use
    `if_false`. Leaves of the tree are bare `NormalizedExpr`s — the tensor's
    value on that sub-region.
    """
    domain : Domain
    if_true : "NormalizedTagTree"
    if_false : "NormalizedTagTree"

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash((self.domain, self.if_true, self.if_false))
        object.__setattr__(self, '_h', h)
        return h


# NormalizedTagTree = NormalizedExpr | NormalizedTagCond
# Defined after NormalizedExpr is declared.


@dataclass(frozen=True)
class NormalizedTensor:
    dims : frozenset[FullDim]
    tag : "NormalizedTagTree | None" = None

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash((self.dims, self.tag))
        object.__setattr__(self, '_h', h)
        return h

@dataclass(frozen=True)
class NormalizedExp:
    child : "NormalizedExpr"

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash(self.child)
        object.__setattr__(self, '_h', h)
        return h

@dataclass(frozen=True)
class NormalizedUnaryOp:
    op : UnaryOpType
    child : "NormalizedExpr"

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash((self.op, self.child))
        object.__setattr__(self, '_h', h)
        return h

@dataclass(frozen=True)
class NormalizedSum:
    children : frozenset["NormalizedProduct"]

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash(self.children)
        object.__setattr__(self, '_h', h)
        return h

@dataclass(frozen=True)
class NormalizedMax:
    children : frozenset["NormalizedExpr"]

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash(self.children)
        object.__setattr__(self, '_h', h)
        return h

@dataclass(frozen=True)
class NormalizedRepeat:
    dims : frozenset[FullDim]
    child : "NormalizedExpr"

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash((self.dims, self.child))
        object.__setattr__(self, '_h', h)
        return h

@dataclass(frozen=True)
class NormalizedReduce:
    """
    `reduce_{j in domain} child(j)` where `j` is the reduction index (a
    `LoopVariable` whose name matches `dim.name`). The `domain` is a DNF
    polyhedron over the reduction index plus any free variables the
    constraints reference (useful for masks that talk about other dims).
    """
    dim : FullDim
    op : ReduceOpType
    domain : Domain
    child : "NormalizedExpr"

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash((self.dim, self.op, self.domain, self.child))
        object.__setattr__(self, '_h', h)
        return h

@dataclass(frozen=True)
class NormalizedParametricReduce:
    """
    A reduction over a loop variable's iteration range: `reduce_{i in [lo, hi)} body(i)`.

    This is the IR primitive for sum/max reductions induced by a rolled loop.
    The loop variable `loop_var` is *bound* inside `body` — any `AffineExpr`
    occurrence of it in `body` refers to this parametric reduction's index,
    not to a free variable in the surrounding scope.
    """
    loop_var : LoopVariable
    lo : SymbolicIndex
    hi : SymbolicIndex
    op : ReduceOpType
    body : "NormalizedExpr"

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash((self.loop_var, self.lo, self.hi, self.op, self.body))
        object.__setattr__(self, '_h', h)
        return h


def _reduce_index(dim : FullDim) -> LoopVariable:
    """
    The `LoopVariable` representing the reduction index over `dim`. Two
    NormalizedReduces that reduce over dims with the same name share this
    index, which is what lets mask predicates reference the same symbolic
    variable across the spec and the normalized IR.
    """
    return LoopVariable(dim.name)


def _bound_sort_key(bound : SymbolicIndex):
    """Stable total order on affine bounds: constant first, then terms by name."""
    aff = to_affine(bound)
    terms = tuple(sorted((v.name, c) for v, c in aff.terms))
    return (aff.const, terms)


def _canonicalize_intervals(
    intervals,
) -> frozenset[tuple[SymbolicIndex, SymbolicIndex]]:
    """
    Merge adjacent 1-D intervals (`a.end == b.start` as affine values).
    Strictly-overlapping intervals are left separate so `sum(a[i:j]) +
    sum(a[k:l])` semantics is preserved (overlaps correctly double-counted).
    Concrete-valued bounds collapse to raw `int`s for canonical equality.
    """
    def concretize(x : SymbolicIndex) -> SymbolicIndex:
        i = as_int(x)
        return i if i is not None else x

    items = [(concretize(s), concretize(e)) for s, e in intervals]
    if not items:
        return frozenset()

    items.sort(key=lambda se: _bound_sort_key(se[0]))
    merged : list[tuple[SymbolicIndex, SymbolicIndex]] = [items[0]]
    for start, end in items[1:]:
        last_start, last_end = merged[-1]
        if to_affine(start) == to_affine(last_end):
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return frozenset(merged)


def _domain_to_intervals(
    domain : Domain,
    dim : FullDim,
) -> frozenset[tuple[SymbolicIndex, SymbolicIndex]] | None:
    """
    If every disjunct in `domain` is a pure 1-D interval over `dim`'s
    reduction index — i.e., exactly two constraints of the form
    `index >= start` and `index < end`, with no other variables in play
    except the reduction index itself — extract the `(start, end)` pairs.
    Otherwise returns None (the domain is truly polyhedral).
    """
    index = _reduce_index(dim)

    def bound_of(expr : AffineExpr, index_coeff : int) -> SymbolicIndex | None:
        """Extract start (coeff=+1) or end-1 (coeff=-1) from a constraint."""
        if _loop_var_coeff(expr, index) != index_coeff:
            return None
        # Isolate the non-index part: `expr - index_coeff * index`.
        index_affine = to_affine(index)
        residue = expr - index_affine * index_coeff
        if _loop_var_coeff(residue, index) != 0:
            return None
        # For `index >= s`, expr = index - s, residue = -s, so start = -residue.
        # For `index < e` encoded as `e - 1 - index >= 0`, expr = -index + (e - 1),
        # residue = e - 1, so end = residue + 1.
        return -residue if index_coeff == 1 else residue + 1

    intervals : set[tuple[SymbolicIndex, SymbolicIndex]] = set()
    for conj in domain.disjuncts:
        if len(conj) != 2:
            return None
        start = None
        end = None
        for c in conj:
            coeff = _loop_var_coeff(c.expr, index)
            if coeff == 1 and start is None:
                start = bound_of(c.expr, 1)
            elif coeff == -1 and end is None:
                end = bound_of(c.expr, -1)
            else:
                return None
        if start is None or end is None:
            return None
        intervals.add((_concretize(start), _concretize(end)))
    return frozenset(intervals)


def make_reduce(
    dim : FullDim,
    op : ReduceOpType,
    intervals,
    child : "NormalizedExpr",
) -> NormalizedReduce:
    """
    Canonical factory for NormalizedReduce. Accepts either a sequence of 1-D
    intervals (will be merged and converted to a Domain) or a ready-made
    Domain. Always use this instead of the raw constructor — the result's
    domain has redundant 1-D bounds collapsed to the tightest pair.
    """
    if isinstance(intervals, Domain):
        domain = intervals
    else:
        merged_intervals = _canonicalize_intervals(intervals)
        domain = interval_domain(_reduce_index(dim), merged_intervals)
    return NormalizedReduce(dim, op, simplify_domain(domain), child)


def _loop_var_coeff(expr : SymbolicIndex, loop_var : LoopVariable) -> int:
    """Coefficient of `loop_var` in an affine expression, or 0 if absent."""
    for v, c in to_affine(expr).terms:
        if v == loop_var:
            return c
    return 0


def _expr_depends_on_var(expr : "NormalizedExpr", loop_var : LoopVariable) -> bool:
    """
    True iff any `SymbolicIndex` bound inside `expr` (e.g. in a Reduce's
    intervals or a Sliced dim in a Tensor) contains `loop_var`.
    """
    def scan_affine(x : SymbolicIndex) -> bool:
        return any(v == loop_var for v, _ in to_affine(x).terms)

    def scan_factor(f) -> bool:
        match f:
            case NormalizedTensor(_):
                return False
            case NormalizedExp(child):
                return _expr_depends_on_var(child, loop_var)
            case NormalizedUnaryOp(_, child):
                return _expr_depends_on_var(child, loop_var)
            case NormalizedSum(children):
                return any(scan_product(c) for c in children)
            case NormalizedMax(children):
                return any(_expr_depends_on_var(c, loop_var) for c in children)
            case NormalizedRepeat(_, child):
                return _expr_depends_on_var(child, loop_var)
            case NormalizedReduce(_, _, domain, child):
                for conj in domain.disjuncts:
                    for constraint in conj:
                        if scan_affine(constraint.expr):
                            return True
                return _expr_depends_on_var(child, loop_var)
            case NormalizedParametricReduce(inner_var, lo, hi, _, body):
                if scan_affine(lo) or scan_affine(hi):
                    return True
                if inner_var == loop_var:
                    # shadowed by inner binder
                    return False
                return _expr_depends_on_var(body, loop_var)
            case _:
                return False

    def scan_product(p : "NormalizedProduct") -> bool:
        return any(scan_factor(f) for f in p.factors)

    return scan_product(expr.num) or scan_product(expr.den)


def _concretize(x : SymbolicIndex) -> SymbolicIndex:
    """If `x` is an AffineExpr with no free variables, return its int value."""
    i = as_int(x)
    return i if i is not None else x


def make_parametric_reduce(
    loop_var : LoopVariable,
    lo : SymbolicIndex,
    hi : SymbolicIndex,
    op : ReduceOpType,
    body : "NormalizedExpr",
) -> "NormalizedExpr":
    """
    Build a parametric reduction and try to collapse it into a single
    `NormalizedReduce` when the body's structure allows. Returns a
    `NormalizedExpr` because the collapsed form usually isn't a single
    factor.

    Recognised patterns:

    1. Empty range (`lo == hi`, concrete): identity — sum returns 0, max
       returns -inf.
    2. Body doesn't depend on `loop_var`: degenerates to a scalar multiple
       (sum → `(hi - lo) * body`) or to `body` itself (max).
    3. Body is exactly a single `NormalizedReduce(dim, op, {(s(k), e(k))})`
       whose interval is affine in `loop_var` with equal `loop_var`
       coefficient in `s` and `e`, adjacent-tile stride (`e(k) - s(k)`
       equals the `loop_var` coefficient), and whose inner child doesn't
       depend on `loop_var`: collapse to `Reduce(dim, op, {(s(lo), e(hi-1))})`.

    Unrecognised patterns fall back to wrapping as `NormalizedParametricReduce`.
    """
    lo, hi = _concretize(lo), _concretize(hi)
    lo_i, hi_i = as_int(lo), as_int(hi)

    # (1) Empty range.
    if lo_i is not None and hi_i is not None and lo_i >= hi_i:
        if op == "sum":
            return NormalizedExpr.of(NormalizedProduct(const=0.0))
        return NormalizedExpr.of(NormalizedProduct(const=float("-inf")))

    # (2) Loop-invariant body.
    if not _expr_depends_on_var(body, loop_var):
        if op == "max":
            return body
        # sum of a loop-invariant `body` over `hi - lo` iterations.
        count = to_affine(hi) - to_affine(lo)
        count_i = as_int(count)
        if count_i is not None:
            scaled = NormalizedProduct(
                body.num.const * count_i, body.num.factors,
            )
            return make_expr(scaled, body.den)
        # Symbolic count: fall through to the wrapped form.

    # (3) Body is a single inner Reduce over an affine-tiled interval.
    collapsed = _try_collapse_tiled_reduce(loop_var, lo, hi, op, body)
    if collapsed is not None:
        return collapsed

    # Fall back: wrap as-is.
    return NormalizedExpr.of(
        NormalizedParametricReduce(loop_var, lo, hi, op, body)
    )


def _try_collapse_tiled_reduce(
    loop_var : LoopVariable,
    lo : SymbolicIndex,
    hi : SymbolicIndex,
    op : ReduceOpType,
    body : "NormalizedExpr",
) -> "NormalizedExpr | None":
    single = _as_single_factor(body.num)
    if (
        body.den.factors
        or body.num.const != 1.0
        or body.den.const != 1.0
        or not isinstance(single, NormalizedReduce)
        or single.op != op
    ):
        return None
    inner_intervals = _domain_to_intervals(single.domain, single.dim)
    if inner_intervals is None or len(inner_intervals) != 1:
        return None
    ((start, end),) = inner_intervals
    start_coeff = _loop_var_coeff(start, loop_var)
    end_coeff = _loop_var_coeff(end, loop_var)
    if start_coeff == 0 or start_coeff != end_coeff:
        return None
    # Tile length must equal stride for adjacency.
    length = to_affine(end) - to_affine(start)
    if length != to_affine(start_coeff):
        return None
    # Inner child mustn't depend on the loop variable.
    if _expr_depends_on_var(single.child, loop_var):
        return None

    # Collapse: union = [start(lo), end(hi - 1)].
    new_start = _substitute_loop_var(start, loop_var, lo)
    new_end = _substitute_loop_var(end, loop_var, to_affine(hi) - 1)
    return NormalizedExpr.of(
        make_reduce(single.dim, op, [(new_start, new_end)], single.child),
    )


def _substitute_loop_var(
    expr : SymbolicIndex,
    loop_var : LoopVariable,
    value : SymbolicIndex,
) -> SymbolicIndex:
    """Replace `loop_var` with `value` in `expr`; returns canonical AffineExpr."""
    aff = to_affine(expr)
    value_aff = to_affine(value)
    result = AffineExpr(aff.const, frozenset())
    for v, c in aff.terms:
        if v == loop_var:
            result = result + value_aff * c
        else:
            result = result + c * v
    return result


def _substitute_lv_in_expr(
    expr : "NormalizedExpr",
    loop_var : LoopVariable,
    value : SymbolicIndex,
) -> "NormalizedExpr":
    """
    Replace every occurrence of `loop_var` inside `expr` with `value`,
    walking the full expression tree (Reduce intervals, nested children,
    etc.). Shadowing by an inner `ParametricReduce` binder is respected.
    """
    return make_expr(
        _substitute_lv_in_product(expr.num, loop_var, value),
        _substitute_lv_in_product(expr.den, loop_var, value),
    )


def _substitute_lv_in_product(
    product : "NormalizedProduct",
    loop_var : LoopVariable,
    value : SymbolicIndex,
) -> "NormalizedProduct":
    new_factors : dict[NormalizedFactor, int] = {}
    for factor, count in product.factors.items():
        new_factor = _substitute_lv_in_factor(factor, loop_var, value)
        new_factors[new_factor] = new_factors.get(new_factor, 0) + count
    return NormalizedProduct(product.const, FrozenCounter.from_dict(new_factors))


def _substitute_lv_in_domain(
    domain : Domain,
    loop_var : LoopVariable,
    value : SymbolicIndex,
) -> Domain:
    """Substitute `loop_var` with `value` throughout every constraint's affine expression."""
    new_disjuncts = frozenset(
        frozenset(
            AffineConstraint(_substitute_loop_var(c.expr, loop_var, value))
            for c in conj
        )
        for conj in domain.disjuncts
    )
    new_vars = frozenset(
        v
        for conj in new_disjuncts
        for c in conj
        for v, _ in c.expr.terms
    )
    return Domain(new_vars, new_disjuncts)


def _substitute_lv_in_factor(
    factor : "NormalizedFactor",
    loop_var : LoopVariable,
    value : SymbolicIndex,
) -> "NormalizedFactor":
    match factor:
        case NormalizedTensor(_):
            return factor
        case NormalizedExp(child):
            return NormalizedExp(_substitute_lv_in_expr(child, loop_var, value))
        case NormalizedUnaryOp(op, child):
            return NormalizedUnaryOp(op, _substitute_lv_in_expr(child, loop_var, value))
        case NormalizedSum(children):
            return NormalizedSum(frozenset(
                _substitute_lv_in_product(c, loop_var, value) for c in children
            ))
        case NormalizedMax(children):
            return NormalizedMax(frozenset(
                _substitute_lv_in_expr(c, loop_var, value) for c in children
            ))
        case NormalizedRepeat(dims, child):
            return NormalizedRepeat(dims, _substitute_lv_in_expr(child, loop_var, value))
        case NormalizedReduce(dim, op, domain, child):
            new_domain = _substitute_lv_in_domain(domain, loop_var, value)
            return make_reduce(
                dim, op, new_domain,
                _substitute_lv_in_expr(child, loop_var, value),
            )
        case NormalizedParametricReduce(inner_var, lo, hi, op, body):
            new_lo = _substitute_loop_var(lo, loop_var, value)
            new_hi = _substitute_loop_var(hi, loop_var, value)
            if inner_var == loop_var:
                # `loop_var` is shadowed inside body by this node's own binder.
                return NormalizedParametricReduce(inner_var, new_lo, new_hi, op, body)
            new_body = _substitute_lv_in_expr(body, loop_var, value)
            return NormalizedParametricReduce(inner_var, new_lo, new_hi, op, new_body)

def _as_single_factor(product : "NormalizedProduct") -> "NormalizedFactor | None":
    """If `product` is exactly one factor with multiplicity one and const=1, return it."""
    if product.const != 1.0 or len(product.factors) != 1:
        return None
    factor, count = next(iter(product.factors.items()))
    return factor if count == 1 else None

def _flatten_sum_terms(terms) -> list["NormalizedProduct"]:
    """Expand any term that is itself a wrapped NormalizedSum into its children."""
    out : list[NormalizedProduct] = []
    for t in terms:
        factor = _as_single_factor(t)
        if isinstance(factor, NormalizedSum):
            out.extend(factor.children)
        else:
            out.append(t)
    return out

def _rebuild_reduce_with_extras(
    dim : FullDim,
    op : ReduceOpType,
    intervals,
    extras : frozenset[AffineConstraint],
    child : "NormalizedExpr",
) -> "NormalizedReduce":
    """
    Build a `NormalizedReduce` whose domain is the union of `intervals`
    over `dim`'s reduce index, intersected with the cross-variable
    `extras` predicate. With `extras` empty this collapses to the plain
    `(intervals, child)` constructor.
    """
    merged_intervals = _canonicalize_intervals(intervals)
    if not extras:
        return make_reduce(dim, op, tuple(merged_intervals), child)
    index = _reduce_index(dim)
    extra_vars = frozenset(v for c in extras for v, _ in c.expr.terms)
    disjuncts : set = set()
    for s, e in merged_intervals:
        conj = set(extras)
        conj.add(ge(index, s))
        conj.add(lt(index, e))
        disjuncts.add(frozenset(conj))
    domain = Domain(frozenset({index}) | extra_vars, frozenset(disjuncts))
    domain = simplify_domain(domain)
    return make_reduce(dim, op, domain, child)


def _split_reduce_domain(
    domain : Domain,
    dim : FullDim,
) -> tuple[
    frozenset[tuple[SymbolicIndex, SymbolicIndex]],
    frozenset[AffineConstraint],
] | None:
    """
    Split a single-disjunct `domain` for a reduce over `dim` into its 1-D
    `(start, end)` interval (constraints purely on the reduction index) and
    a frozenset of "extra" cross-variable constraints (e.g., a `where`-clause
    predicate on the reduce dim and other dims). Returns None when the
    domain isn't single-disjunct or doesn't have a clean 1-D bound pair on
    the reduction index.
    """
    if len(domain.disjuncts) != 1:
        return None
    conj = next(iter(domain.disjuncts))
    index = _reduce_index(dim)

    start = None
    end = None
    extras : list[AffineConstraint] = []
    for c in conj:
        coeff = _loop_var_coeff(c.expr, index)
        # Only treat a constraint as a 1-D bound on the reduction index if its
        # residue (after subtracting the index term) is a plain constant —
        # otherwise we'd treat e.g. `nctx <= qctx` as `end = qctx+1`, and the
        # interval-merge across tiles would never converge because the bound
        # carries a free variable. Cross-variable constraints go in `extras`.
        if coeff == 1 and start is None:
            residue = c.expr - to_affine(index)
            if residue.terms:
                extras.append(c)
                continue
            start = -residue
        elif coeff == -1 and end is None:
            residue = c.expr + to_affine(index)
            if residue.terms:
                extras.append(c)
                continue
            end = residue + 1
        else:
            extras.append(c)
    if start is None or end is None:
        return None
    interval = (_concretize(start), _concretize(end))
    return frozenset({interval}), frozenset(extras)


def _merge_sum_reduces(terms : list["NormalizedProduct"]) -> list["NormalizedProduct"]:
    """
    Combine `const * sum-Reduce(...)` terms that share const, dim, child,
    and any extra cross-variable constraints (e.g., a `where`-clause
    predicate). The reduce's 1-D bounds on the reduction index are unioned
    interval-style; the extras are preserved verbatim. This is what lets
    `(sum[a:b] f where P) + (sum[b:c] f where P)` collapse to
    `sum[a:c] f where P` even when `P` references variables outside the
    reduction.
    """
    groups : dict[
        tuple[float, FullDim, "NormalizedExpr", frozenset[AffineConstraint]],
        list[tuple[SymbolicIndex, SymbolicIndex]],
    ] = {}
    others : list[NormalizedProduct] = []
    for t in terms:
        factor = _as_single_factor(
            NormalizedProduct(const=1.0, factors=t.factors),
        )
        if isinstance(factor, NormalizedReduce) and factor.op == "sum":
            split = _split_reduce_domain(factor.domain, factor.dim)
            if split is not None:
                intervals, extras = split
                key = (t.const, factor.dim, factor.child, extras)
                groups.setdefault(key, []).extend(intervals)
                continue
        others.append(t)

    # NOTE: implicit-zero-bias promotion (folding an untagged group into
    # a tagged sibling) intentionally not implemented here — it requires
    # polyhedral subsumption against the *actual* outer-slice ranges of
    # the predicate's free variables, and those ranges live in the
    # surrounding `Type.st` which isn't visible to the post-hoc
    # normalizer. Tracked as task #6 (blocked on task #8: thread outer
    # slice constraints into reduce domains).

    merged : list[NormalizedProduct] = []
    for (const, dim, child, extras), intervals in groups.items():
        reduce_factor = _rebuild_reduce_with_extras(dim, "sum", intervals, extras, child)
        merged.append(NormalizedProduct(
            const=const,
            factors=FrozenCounter.from_iterable([reduce_factor]),
        ))

    return others + merged

def _is_pure_const(expr : "NormalizedExpr") -> bool:
    return not expr.num.factors and not expr.den.factors

def _pure_parametric_factor(c : "NormalizedExpr", op : ReduceOpType):
    """
    If `c` is exactly a wrapped `NormalizedParametricReduce` with the given
    `op` (const=1, single factor with count 1, empty denominator), return
    the parametric factor. Else None.
    """
    if c.den.factors or c.num.const != 1.0 or c.den.const != 1.0:
        return None
    f = _as_single_factor(c.num)
    if isinstance(f, NormalizedParametricReduce) and f.op == op:
        return f
    return None


def _absorb_max_boundaries(children : list["NormalizedExpr"]) -> list["NormalizedExpr"]:
    """
    `max(ParametricMax(k in [lo, hi), f(k)), f(hi))` → `ParametricMax(k in [lo, hi+1), f(k))`.
    And symmetrically at the lower bound.
    """
    parametrics, others = [], []
    for c in children:
        (parametrics if _pure_parametric_factor(c, "max") else others).append(c)
    if not parametrics:
        return children

    result : list[NormalizedExpr] = []
    for p in parametrics:
        pf = _pure_parametric_factor(p, "max")
        loop_var, lo, hi, body = pf.loop_var, pf.lo, pf.hi, pf.body
        changed = True
        while changed and others:
            changed = False
            expected_hi = _substitute_lv_in_expr(body, loop_var, hi)
            for i, sib in enumerate(others):
                if sib == expected_hi:
                    hi = to_affine(hi) + 1
                    others.pop(i)
                    changed = True
                    break
            if changed:
                continue
            expected_lo = _substitute_lv_in_expr(body, loop_var, to_affine(lo) - 1)
            for i, sib in enumerate(others):
                if sib == expected_lo:
                    lo = to_affine(lo) - 1
                    others.pop(i)
                    changed = True
                    break
        result.append(make_parametric_reduce(loop_var, lo, hi, "max", body))
    return others + result


def make_max(children) -> "NormalizedExpr":
    """Build a canonical NormalizedMax. Flattens nested maxes, drops -inf terms
    (identity for max), merges max-Reduce children that share dim and child by
    unioning their intervals, absorbs boundary terms into ParametricMax, dedupes,
    and collapses singletons.
    """
    flat : list[NormalizedExpr] = []
    for c in children:
        single = None
        if not c.den.factors and c.num.const == 1.0 and c.den.const == 1.0:
            single = _as_single_factor(c.num)
        if isinstance(single, NormalizedMax):
            flat.extend(single.children)
        else:
            flat.append(c)

    # Drop -inf children — they are the identity for max.
    flat = [c for c in flat if not (_is_pure_const(c) and c.num.const == float("-inf"))]

    # Group max-Reduce children by (dim, body, cross-variable extras) so
    # tagged-with-shared-predicate max-reduces fuse alongside the plain
    # 1-D-bound case. Mirror of `_merge_sum_reduces`.
    groups : dict[
        tuple[FullDim, NormalizedExpr, frozenset[AffineConstraint]],
        list[tuple[SymbolicIndex, SymbolicIndex]],
    ] = {}
    others : list[NormalizedExpr] = []
    for c in flat:
        single = None
        if not c.den.factors and c.num.const == 1.0 and c.den.const == 1.0:
            single = _as_single_factor(c.num)
        if isinstance(single, NormalizedReduce) and single.op == "max":
            split = _split_reduce_domain(single.domain, single.dim)
            if split is not None:
                intervals, extras = split
                key = (single.dim, single.child, extras)
                groups.setdefault(key, []).extend(intervals)
                continue
        others.append(c)

    for (d, child, extras), intervals in groups.items():
        merged = _rebuild_reduce_with_extras(d, "max", intervals, extras, child)
        others.append(NormalizedExpr.of(merged))

    others = _absorb_max_boundaries(others)

    unique = frozenset(others)
    if not unique:
        return NormalizedExpr.of(NormalizedProduct(const=float("-inf")))
    if len(unique) == 1:
        return next(iter(unique))
    return NormalizedExpr.of(NormalizedMax(unique))

def _absorb_sum_boundaries(terms : list["NormalizedProduct"]) -> list["NormalizedProduct"]:
    """
    `ParametricSum(k in [lo, hi), f(k)) + f(hi)` → `ParametricSum(k in [lo, hi+1), f(k))`.
    And symmetrically at the lower bound.
    """
    parametrics_and_others : list[tuple[bool, NormalizedProduct]] = []
    for t in terms:
        wrapped = NormalizedExpr.of(NormalizedProduct(const=1.0, factors=t.factors))
        pf = _pure_parametric_factor(wrapped, "sum") if t.const == 1.0 else None
        parametrics_and_others.append((pf is not None, t))

    parametrics = [t for is_p, t in parametrics_and_others if is_p]
    others = [t for is_p, t in parametrics_and_others if not is_p]
    if not parametrics:
        return terms

    result : list[NormalizedProduct] = []
    for p in parametrics:
        wrapped = NormalizedExpr.of(NormalizedProduct(const=1.0, factors=p.factors))
        pf = _pure_parametric_factor(wrapped, "sum")
        loop_var, lo, hi, body = pf.loop_var, pf.lo, pf.hi, pf.body
        changed = True
        while changed and others:
            changed = False
            expected_hi = _substitute_lv_in_expr(body, loop_var, hi)
            for i, sib in enumerate(others):
                if sib.const == 1.0:
                    sib_expr = NormalizedExpr.of(NormalizedProduct(1.0, sib.factors))
                    if sib_expr == expected_hi:
                        hi = to_affine(hi) + 1
                        others.pop(i)
                        changed = True
                        break
            if changed:
                continue
            expected_lo = _substitute_lv_in_expr(body, loop_var, to_affine(lo) - 1)
            for i, sib in enumerate(others):
                if sib.const == 1.0:
                    sib_expr = NormalizedExpr.of(NormalizedProduct(1.0, sib.factors))
                    if sib_expr == expected_lo:
                        lo = to_affine(lo) - 1
                        others.pop(i)
                        changed = True
                        break
        # Rebuild into a NormalizedProduct (wrapping the possibly-collapsed
        # parametric-reduce result).
        param_expr = make_parametric_reduce(loop_var, lo, hi, "sum", body)
        # Fold the expr back into a product term; assumes param_expr is a
        # singleton-factor expression (true for collapsed-Reduce or parametric).
        if param_expr.den.factors or param_expr.den.const != 1.0:
            # Can't cleanly express a fractional result as a sum term; skip absorption.
            result.append(p)
            continue
        result.append(
            NormalizedProduct(param_expr.num.const, param_expr.num.factors)
        )
    return others + result


def make_sum(terms) -> "NormalizedProduct":
    """Build a canonical sum-of-products, returned as a NormalizedProduct.
    Flattens nested NormalizedSums, drops zero terms, combines like terms,
    merges disjoint sum-Reduce intervals, absorbs boundary terms into
    ParametricSum, extracts common factors, and collapses singletons.
    """
    flat = _flatten_sum_terms(terms)
    flat = [t for t in flat if t.const != 0.0]

    # -inf absorption: if any term is a pure -inf constant, the whole sum is -inf.
    for t in flat:
        if t.const == float("-inf") and not t.factors:
            return NormalizedProduct(const=float("-inf"))

    flat = _merge_sum_reduces(flat)
    flat = _absorb_sum_boundaries(flat)

    by_factors : dict[FrozenCounter, float] = {}
    for t in flat:
        by_factors[t.factors] = by_factors.get(t.factors, 0.0) + t.const
    combined = [
        NormalizedProduct(const=c, factors=f)
        for f, c in by_factors.items()
        if c != 0.0
    ]

    if not combined:
        return NormalizedProduct(const=0.0)
    if len(combined) == 1:
        return combined[0]

    # Extract factors common to every term: A*X + A*Y = A*(X+Y).
    common = combined[0].factors
    for t in combined[1:]:
        common = common & t.factors
    if common:
        inner = make_sum([
            NormalizedProduct(t.const, t.factors - common)
            for t in combined
        ])
        return NormalizedProduct(inner.const, common + inner.factors)

    return NormalizedProduct(
        const=1.0,
        factors=FrozenCounter.from_iterable([
            NormalizedSum(frozenset(combined)),
        ]),
    )

NormalizedFactor = NormalizedTensor | NormalizedExp | NormalizedUnaryOp | NormalizedSum | NormalizedMax | NormalizedRepeat | NormalizedReduce | NormalizedParametricReduce
# NormalizedTagTree is defined after NormalizedExpr is declared (below).

@dataclass(frozen=True)
class NormalizedProduct:
    const : float = 1.0
    factors : FrozenCounter["NormalizedFactor"] = field(default_factory=FrozenCounter.empty)

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash((self.const, self.factors))
        object.__setattr__(self, '_h', h)
        return h

    def has_repeat(self) -> bool:
        cached = getattr(self, '_hr', None)
        if cached is not None:
            return cached
        result = False
        for f in self.factors:
            if isinstance(f, NormalizedRepeat):
                result = True
                break
        object.__setattr__(self, '_hr', result)
        return result

@dataclass(frozen=True)
class NormalizedExpr:
    num : NormalizedProduct
    den : NormalizedProduct

    def __post_init__(self):
        assert self.den.const == 1.0

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash((self.num, self.den))
        object.__setattr__(self, '_h', h)
        return h

    @staticmethod
    def of(x : "NormalizedExpr | NormalizedProduct | NormalizedFactor") -> "NormalizedExpr":
        if isinstance(x, NormalizedExpr):
            return x
        empty = NormalizedProduct()
        if isinstance(x, NormalizedProduct):
            if not x.has_repeat():
                return NormalizedExpr(x, empty)
            return make_expr(x, empty)
        # x is a NormalizedFactor
        num = NormalizedProduct(factors=FrozenCounter.from_iterable([x]))
        if not isinstance(x, NormalizedRepeat):
            return NormalizedExpr(num, empty)
        return make_expr(num, empty)


NormalizedTagTree = NormalizedExpr | NormalizedTagCond


def _hoist_invariants_from_repeat(
    repeat : NormalizedRepeat,
) -> tuple[NormalizedProduct, NormalizedProduct, "NormalizedRepeat | None"]:
    """Apply `Repeat(D, A_invar * G_vary / (B_invar * H_vary)) =
    A_invar / B_invar * Repeat(D, G_vary / H_vary)`. Returns
    (outer numerator product, outer denominator product, simplified repeat or None).
    If the Repeat's varying part is empty, the Repeat dissolves entirely and its
    leftover constant is baked into the outer numerator product.
    """
    dims = repeat.dims
    child = repeat.child

    def split(factors : FrozenCounter[NormalizedFactor]) -> tuple[
        dict[NormalizedFactor, int], dict[NormalizedFactor, int]
    ]:
        invariant : dict[NormalizedFactor, int] = {}
        varying : dict[NormalizedFactor, int] = {}
        for f, c in factors.items():
            if all(not varies_with_dim(f, d) for d in dims):
                invariant[f] = c
            else:
                varying[f] = c
        return invariant, varying

    num_invariant, num_varying = split(child.num.factors)
    den_invariant, den_varying = split(child.den.factors)

    new_child_num = NormalizedProduct(child.num.const, FrozenCounter.from_dict(num_varying))
    new_child_den = NormalizedProduct(1.0, FrozenCounter.from_dict(den_varying))

    if not new_child_num.factors and not new_child_den.factors:
        # The Repeat is over a pure constant (or nothing): it dissolves. Bake the
        # leftover const into the outer numerator product.
        outer_num_prod = NormalizedProduct(
            new_child_num.const,
            FrozenCounter.from_dict(num_invariant),
        )
        outer_den_prod = NormalizedProduct(1.0, FrozenCounter.from_dict(den_invariant))
        return outer_num_prod, outer_den_prod, None

    outer_num_prod = NormalizedProduct(1.0, FrozenCounter.from_dict(num_invariant))
    outer_den_prod = NormalizedProduct(1.0, FrozenCounter.from_dict(den_invariant))
    new_child = NormalizedExpr(new_child_num, new_child_den)
    return outer_num_prod, outer_den_prod, NormalizedRepeat(dims, new_child)

def make_expr(num : NormalizedProduct, den : NormalizedProduct) -> NormalizedExpr:
    """Canonical factory: cancels common factors between numerator and denominator,
    hoists Repeat(D, ...)-invariant factors out of Repeats, enforces den.const == 1.0,
    and drops factors from a numerator whose const is 0 (zero annihilates).
    """
    # 0 * anything = 0 and 0 / anything = 0: canonicalize so both the
    # num's factors and the den drop, otherwise `0 / x` and `0` are
    # structurally distinct even though they're algebraically equal.
    if num.const == 0.0:
        return NormalizedExpr(
            NormalizedProduct(const=0.0, factors=FrozenCounter.empty()),
            NormalizedProduct(const=1.0, factors=FrozenCounter.empty()),
        )

    # Fast path: no NormalizedRepeats anywhere, so the hoist step is a no-op.
    # Just cancel common factors and return.
    if not num.has_repeat() and not den.has_repeat():
        if num.factors and den.factors:
            common = num.factors & den.factors
            if common:
                num = NormalizedProduct(num.const, num.factors - common)
                den = NormalizedProduct(den.const, den.factors - common)
        return NormalizedExpr(num, den)

    # Hoist invariant factors out of Repeats in both num and den. Repeats in num
    # contribute their pulled num/den to the outer num/den; Repeats in den flip sides.
    num_const = num.const
    den_const_inverse = 1.0
    num_extras_num : dict[NormalizedFactor, int] = {}
    num_extras_den : dict[NormalizedFactor, int] = {}
    new_num_factors : dict[NormalizedFactor, int] = {}
    for f, c in num.factors.items():
        if isinstance(f, NormalizedRepeat):
            pulled_num, pulled_den, new_rep = _hoist_invariants_from_repeat(f)
            num_const *= pulled_num.const ** c
            den_const_inverse *= pulled_den.const ** c
            for pf, pc in pulled_num.factors.items():
                num_extras_num[pf] = num_extras_num.get(pf, 0) + pc * c
            for pf, pc in pulled_den.factors.items():
                num_extras_den[pf] = num_extras_den.get(pf, 0) + pc * c
            if new_rep is not None:
                new_num_factors[new_rep] = new_num_factors.get(new_rep, 0) + c
        else:
            new_num_factors[f] = new_num_factors.get(f, 0) + c

    den_extras_num : dict[NormalizedFactor, int] = {}
    den_extras_den : dict[NormalizedFactor, int] = {}
    new_den_factors : dict[NormalizedFactor, int] = {}
    for f, c in den.factors.items():
        if isinstance(f, NormalizedRepeat):
            pulled_num, pulled_den, new_rep = _hoist_invariants_from_repeat(f)
            # Repeat in den: its num-side pullups go to outer den, den-side go to outer num.
            num_const /= pulled_num.const ** c
            den_const_inverse *= pulled_den.const ** c
            for pf, pc in pulled_num.factors.items():
                den_extras_den[pf] = den_extras_den.get(pf, 0) + pc * c
            for pf, pc in pulled_den.factors.items():
                den_extras_num[pf] = den_extras_num.get(pf, 0) + pc * c
            if new_rep is not None:
                new_den_factors[new_rep] = new_den_factors.get(new_rep, 0) + c
        else:
            new_den_factors[f] = new_den_factors.get(f, 0) + c

    num = NormalizedProduct(
        num_const * den_const_inverse,
        FrozenCounter.from_dict(new_num_factors)
            + FrozenCounter.from_dict(num_extras_num)
            + FrozenCounter.from_dict(den_extras_num),
    )
    den = NormalizedProduct(
        1.0,
        FrozenCounter.from_dict(new_den_factors)
            + FrozenCounter.from_dict(num_extras_den)
            + FrozenCounter.from_dict(den_extras_den),
    )

    # Cancel common factors.
    common = num.factors & den.factors
    if common:
        num = NormalizedProduct(num.const, num.factors - common)
        den = NormalizedProduct(den.const, den.factors - common)
    return NormalizedExpr(num, den)

def _single_tagged_tensor(nexpr : NormalizedExpr) -> "NormalizedTensor | None":
    """
    If `nexpr` is exactly `Expr.of(tagged_tensor)` (const=1, one factor with
    count 1, no denominator), return the tagged tensor. Otherwise None.
    Used to detect when a binary/unary op can be distributed through a tag.
    """
    if (
        nexpr.num.const != 1.0
        or nexpr.den.factors
        or nexpr.den.const != 1.0
    ):
        return None
    factor = _as_single_factor(nexpr.num)
    if isinstance(factor, NormalizedTensor) and factor.tag is not None:
        return factor
    return None


def _push_through_tag(
    transform,
    tensor : "NormalizedTensor",
) -> NormalizedExpr:
    """
    Apply `transform(NormalizedExpr) -> NormalizedExpr` to every leaf of
    `tensor.tag`, returning a new tagged tensor wrapped in an Expr. The
    generic rule for pushing any op through a piecewise structure.
    """
    def apply(tag):
        if isinstance(tag, NormalizedTagCond):
            return NormalizedTagCond(
                domain=tag.domain,
                if_true=apply(tag.if_true),
                if_false=apply(tag.if_false),
            )
        return transform(tag)
    return NormalizedExpr.of(
        NormalizedTensor(dims=tensor.dims, tag=apply(tensor.tag))
    )


def _distribute_binop_through_tag(
    op_fn,
    lhs : NormalizedExpr,
    rhs : NormalizedExpr,
) -> NormalizedExpr | None:
    """
    If exactly one of lhs/rhs is a singleton tagged-tensor expression,
    distribute `op_fn` through its tag branches — the generic rule for
    `op(tagged, x) = tagged(D, op(t, x), op(f, x))`. If both are tagged,
    nest. If neither, return None.
    """
    lhs_tag = _single_tagged_tensor(lhs)
    rhs_tag = _single_tagged_tensor(rhs)
    if lhs_tag is None and rhs_tag is None:
        return None
    if lhs_tag is not None and rhs_tag is None:
        return _push_through_tag(lambda leaf: op_fn(leaf, rhs), lhs_tag)
    if lhs_tag is None and rhs_tag is not None:
        return _push_through_tag(lambda leaf: op_fn(lhs, leaf), rhs_tag)
    # Both tagged: nest (outer = lhs's tag, inner = rhs's tag).
    return _push_through_tag(
        lambda l_leaf: _push_through_tag(
            lambda r_leaf: op_fn(l_leaf, r_leaf), rhs_tag,
        ),
        lhs_tag,
    )


def add(lhs : NormalizedExpr, rhs : NormalizedExpr):
    distributed = _distribute_binop_through_tag(add, lhs, rhs)
    if distributed is not None:
        return distributed
    a, b = lhs.num, lhs.den
    c, d = rhs.num, rhs.den

    # (a/b) + (c/d) = (ad + bc) / (bd)

    # Note that regarding the constants, since we have the invariant
    # that the const in the denominator is always 1.0, the constants just
    # end up being a.const and c.const

    ad = NormalizedProduct(
        const=a.const,
        factors=a.factors + d.factors,
    )
    bc = NormalizedProduct(
        const=c.const,
        factors=b.factors + c.factors,
    )
    bd = NormalizedProduct(
        const=1.0,
        factors=b.factors + d.factors,
    )
    return make_expr(make_sum([ad, bc]), bd)

def sub(lhs : NormalizedExpr, rhs : NormalizedExpr):
    distributed = _distribute_binop_through_tag(sub, lhs, rhs)
    if distributed is not None:
        return distributed
    a, b = lhs.num, lhs.den
    c, d = rhs.num, rhs.den

    # (a/b) - (c/d) = (ad - bc) / (bd)
    # Note that regarding the constants, since we have the invariant
    # that the const in the denominator is always 1.0, the constants just
    # end up being a.const and -c.const

    ad = NormalizedProduct(
        const=a.const,
        factors=a.factors + d.factors,
    )
    bc = NormalizedProduct(
        const=-c.const,
        factors=b.factors + c.factors,
    )
    bd = NormalizedProduct(
        const=1.0,
        factors=b.factors + d.factors,
    )
    return make_expr(make_sum([ad, bc]), bd)

def mul(lhs : NormalizedExpr, rhs : NormalizedExpr):
    distributed = _distribute_binop_through_tag(mul, lhs, rhs)
    if distributed is not None:
        return distributed
    a, b = lhs.num, lhs.den
    c, d = rhs.num, rhs.den

    # (a/b) * (c/d) = (ac)/(bd)
    ac = NormalizedProduct(
        const=(a.const * c.const) / (b.const * d.const),
        factors=a.factors + c.factors,
    )
    bd = NormalizedProduct(
        const=1.0,
        factors=b.factors + d.factors,
    )
    return make_expr(ac, bd)

def div(lhs : NormalizedExpr, rhs : NormalizedExpr):
    distributed = _distribute_binop_through_tag(div, lhs, rhs)
    if distributed is not None:
        return distributed
    a, b = lhs.num, lhs.den
    c, d = rhs.num, rhs.den

    # (a/b) / (c/d) = (ad)/(bc)
    ad = NormalizedProduct(
        const=(a.const * d.const) / (b.const * c.const),
        factors=a.factors + d.factors,
    )
    bc = NormalizedProduct(
        const=1.0,
        factors=b.factors + c.factors,
    )
    return make_expr(ad, bc)


def normalize_exp(nchild : NormalizedExpr) -> NormalizedExpr:
    """Distribute exp across a sum: exp(a + b - c) = exp(a)*exp(b)/exp(c).
    Collapse exp(c) = math.exp(c) for any pure-constant c — covers
    `exp(0) = 1`, `exp(-inf) = 0`, etc. Push exp through a top-level
    tagged tensor so bias-form `Cond(D, 0, -inf)` lands at mask-form
    `Cond(D, 1, 0)`. Otherwise wrap as NormalizedExp.
    """
    if _is_pure_const(nchild):
        return NormalizedExpr.of(
            NormalizedProduct(const=math.exp(nchild.num.const))
        )

    pushed = _push_unary_through_tag_expr("exp", nchild)
    if pushed is not None:
        return pushed

    sum_factor = None
    if (
        nchild.num.const == 1.0
        and nchild.den.const == 1.0
        and not nchild.den.factors
    ):
        single = _as_single_factor(nchild.num)
        if isinstance(single, NormalizedSum):
            sum_factor = single

    if sum_factor is None:
        return NormalizedExpr.of(NormalizedExp(nchild))

    num_factors : dict[NormalizedFactor, int] = {}
    den_factors : dict[NormalizedFactor, int] = {}
    for term in sum_factor.children:
        term_expr = NormalizedExpr.of(
            NormalizedProduct(abs(term.const), term.factors)
        )
        target = num_factors if term.const >= 0 else den_factors

        # Push exp through any tagged tensor leaf — `exp(-inf) = 0` and
        # `exp(0) = 1` constant-fold at the leaf, so a bias `Cond(D, 0, -inf)`
        # lands naturally at the mask `Cond(D, 1, 0)`.
        if term.const >= 0:
            pushed = _push_unary_through_tag_expr("exp", term_expr)
            if pushed is not None and _single_tagged_tensor(pushed) is not None:
                tagged_factor, cnt = next(iter(pushed.num.factors.items()))
                target[tagged_factor] = target.get(tagged_factor, 0) + cnt
                continue

        exp_factor = NormalizedExp(term_expr)
        target[exp_factor] = target.get(exp_factor, 0) + 1

    num = NormalizedProduct(1.0, FrozenCounter.from_dict(num_factors))
    den = NormalizedProduct(1.0, FrozenCounter.from_dict(den_factors))
    return make_expr(num, den)


def _push_unary_through_tag_expr(
    op : UnaryOpType,
    nchild : NormalizedExpr,
) -> NormalizedExpr | None:
    """
    If `nchild` is a single tagged-tensor factor, push `op` through each
    leaf of its tag; otherwise return None.
    """
    tagged = _single_tagged_tensor(nchild)
    if tagged is None:
        return None
    return _push_through_tag(lambda leaf: unary_op(op, leaf), tagged)


_SCALAR_UNARY_OPS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "exp": math.exp,
}

def unary_op(op : UnaryOpType, nchild : NormalizedExpr) -> NormalizedExpr:
    if op == "exp":
        return normalize_exp(nchild)
    if _is_pure_const(nchild) and op in _SCALAR_UNARY_OPS:
        return NormalizedExpr.of(NormalizedProduct(const=_SCALAR_UNARY_OPS[op](nchild.num.const)))
    # Push unary op through a tagged-tensor input so constant-folding at
    # the tag's leaves converts structured values (bias/mask/etc.) directly.
    pushed = _push_unary_through_tag_expr(op, nchild)
    if pushed is not None:
        return pushed
    return NormalizedExpr.of(NormalizedUnaryOp(op, nchild))

def binary_op(op : BinaryOpType, nlhs : NormalizedExpr, nrhs : NormalizedExpr) -> NormalizedExpr:
    match op:
        case "+":
            return add(nlhs, nrhs)
        case "-":
            return sub(nlhs, nrhs)
        case "*":
            return mul(nlhs, nrhs)
        case "/":
            return div(nlhs, nrhs)
        case "max":
            return make_max([nlhs, nrhs])

def repeat(dims : frozenset[FullDim], x : ExprType) -> NormalizedExpr:
    match x:
        case Constant(_):
            # Pure scalar constants broadcast implicitly; no need to wrap in NormalizedRepeat.
            return normalize(x)
        case UnaryOp(op, child):
            rchild = repeat(dims, child)
            return unary_op(op, rchild)
        case BinaryOp(op, lhs, rhs):
            rlhs = repeat(dims, lhs)
            rrhs = repeat(dims, rhs)
            return binary_op(op, rlhs, rrhs)
        case Repeat(dim, child):
            return repeat(dims | { dim }, child)
        case _:
            child_normalized = normalize(x)
            if _is_pure_const(child_normalized):
                return child_normalized
            # Push Repeat through tags: Repeat(D, Cond(P, a, b)) = Cond(P, Repeat(D, a), Repeat(D, b)),
            # with the tagged tensor's dims growing by `dims`. For scalar
            # leaves (masks, biases), Repeat is a no-op and leaves are
            # untouched — the tensor just claims a larger broadcast-shape.
            tagged = _single_tagged_tensor(child_normalized)
            if tagged is not None:
                def push(leaf):
                    if _is_pure_const(leaf):
                        return leaf
                    return NormalizedExpr.of(NormalizedRepeat(dims, leaf))
                def apply(tag):
                    if isinstance(tag, NormalizedTagCond):
                        return NormalizedTagCond(
                            domain=tag.domain,
                            if_true=apply(tag.if_true),
                            if_false=apply(tag.if_false),
                        )
                    return push(tag)
                return NormalizedExpr.of(
                    NormalizedTensor(dims=tagged.dims | dims, tag=apply(tagged.tag))
                )
            return NormalizedExpr.of(NormalizedRepeat(dims, child_normalized))

def varies_with_dim(e : NormalizedFactor | NormalizedProduct | NormalizedExpr, dim : FullDim) -> bool:
    """True iff the value of `e` changes as you move along `dim`.

    Repeat(dim, x) is *constant* along dim (broadcasts a single value) — returns False.
    Reduce(dim, x) eliminates dim, so the result is also constant along dim — returns False.
    """
    match e:
        case NormalizedTensor(dims):
            return dim in dims
        case NormalizedExp(child):
            return varies_with_dim(child, dim)
        case NormalizedUnaryOp(_, child):
            return varies_with_dim(child, dim)
        case NormalizedSum(children):
            return any(varies_with_dim(c, dim) for c in children)
        case NormalizedMax(children):
            return any(varies_with_dim(c, dim) for c in children)
        case NormalizedRepeat(dims, child):
            if dim in dims:
                return False
            return varies_with_dim(child, dim)
        case NormalizedReduce(d, _, _, child):
            if d == dim:
                return False
            return varies_with_dim(child, dim)
        case NormalizedProduct(_, factors):
            return any(varies_with_dim(f, dim) for f in factors)
        case NormalizedExpr(num, den):
            return varies_with_dim(num, dim) or varies_with_dim(den, dim)


def partition_by_dim_variance(
    dim : FullDim,
    expr : NormalizedProduct,
) -> tuple[FrozenCounter[NormalizedFactor], FrozenCounter[NormalizedFactor]]:
    """Split a product's factors into (varying, invariant) with respect to `dim`.

    Varying factors must stay inside a reduce over `dim`; invariant factors can be
    hoisted out.
    """
    varying : dict[NormalizedFactor, int] = {}
    invariant : dict[NormalizedFactor, int] = {}

    for factor, count in expr.factors.items():
        if varies_with_dim(factor, dim):
            varying[factor] = count
        else:
            invariant[factor] = count

    return FrozenCounter.from_dict(varying), FrozenCounter.from_dict(invariant)


def strip_repeats_from_factor(factor : NormalizedFactor, dim : FullDim) -> NormalizedExpr:
    """Recursively eliminate Repeat(dim, ...) wrappers anywhere inside `factor`.

    Precondition: `factor` is dim-invariant (i.e., `not varies_with_dim(factor, dim)`).
    A `Repeat(dim, inner)` wrapper is redundant once the enclosing context no longer
    has `dim` in its shape (e.g., after being hoisted out of a sum-reduce over `dim`).

    Returns a NormalizedExpr because a dissolved `NormalizedRepeat` may contribute
    arbitrary num/den structure that can't fit back into a single factor slot.
    """
    match factor:
        case NormalizedTensor(_):
            return NormalizedExpr.of(factor)
        case NormalizedExp(child):
            return NormalizedExpr.of(NormalizedExp(strip_repeats_from_expr(child, dim)))
        case NormalizedUnaryOp(op, child):
            return NormalizedExpr.of(NormalizedUnaryOp(op, strip_repeats_from_expr(child, dim)))
        case NormalizedSum(children):
            new_children = frozenset(
                c  # sum children are products; stripping a product returns an expr,
                   # but invariant-sum children shouldn't grow denominators. Punt: recurse
                   # via product, then assume back to product form.
                for c in children
            )
            # For now recurse shallowly: if sum's children were invariant, stripping
            # wouldn't introduce denominators. Use a light recursion.
            return NormalizedExpr.of(NormalizedSum(frozenset(
                _strip_product_keeping_product(c, dim) for c in children
            )))
        case NormalizedMax(children):
            return NormalizedExpr.of(NormalizedMax(frozenset(
                strip_repeats_from_expr(c, dim) for c in children
            )))
        case NormalizedRepeat(dims, child):
            stripped_child = strip_repeats_from_expr(child, dim)
            new_dims = dims - { dim }
            if new_dims:
                return NormalizedExpr.of(NormalizedRepeat(new_dims, stripped_child))
            return stripped_child
        case NormalizedReduce(d, op, domain, child):
            return NormalizedExpr.of(
                make_reduce(d, op, domain, strip_repeats_from_expr(child, dim))
            )

def _strip_product_keeping_product(product : NormalizedProduct, dim : FullDim) -> NormalizedProduct:
    """Shallow strip that preserves the NormalizedProduct type. Used for sum-children,
    which must remain products."""
    new_factors : dict[NormalizedFactor, int] = {}
    for factor, count in product.factors.items():
        stripped = strip_repeats_from_factor(factor, dim)
        single = _as_single_factor(stripped.num) if not stripped.den.factors and stripped.den.const == 1.0 and stripped.num.const == 1.0 else None
        if single is not None:
            new_factors[single] = new_factors.get(single, 0) + count
        else:
            # Can't represent cleanly as a single factor — skip stripping this one.
            new_factors[factor] = new_factors.get(factor, 0) + count
    return NormalizedProduct(product.const, FrozenCounter.from_dict(new_factors))

def strip_repeats_from_product(product : NormalizedProduct, dim : FullDim) -> NormalizedExpr:
    """Strip Repeats of `dim` from every factor in `product` and fold the result
    back into a single NormalizedExpr. Assumes all factors are dim-invariant."""
    result = NormalizedExpr.of(NormalizedProduct(const=product.const))
    for factor, count in product.factors.items():
        stripped = strip_repeats_from_factor(factor, dim)
        for _ in range(count):
            result = mul(result, stripped)
    return result

def strip_repeats_from_expr(expr : NormalizedExpr, dim : FullDim) -> NormalizedExpr:
    num_expr = strip_repeats_from_product(expr.num, dim)
    den_expr = strip_repeats_from_product(expr.den, dim)
    return div(num_expr, den_expr)


def _extract_tagged_body(
    expr : NormalizedExpr,
    op : ReduceOpType,
) -> tuple[NormalizedExpr, Domain] | None:
    """
    If `expr` is a single tagged tensor with shape `Cond(D, body, identity)`
    — where `identity` is the op's reduction identity (0 for sum, -inf for
    max) — return `(body, D)`. Caller then uses `body` as the reduce's body
    and intersects its domain with `D`.

    Under the generic tag push-through, `body * mask(D)` becomes
    `Cond(D, body, 0)` and `exp(scores + bias(D))` becomes
    `Cond(D, exp(scores), 0)`, both of which this extractor folds back into
    a domain-restricted reduce.
    """
    tagged = _single_tagged_tensor(expr)
    if tagged is None or not isinstance(tagged.tag, NormalizedTagCond):
        return None
    cond = tagged.tag
    identity = 0.0 if op == "sum" else float("-inf")
    if (
        _is_pure_const(cond.if_false)
        and cond.if_false.num.const == identity
    ):
        return cond.if_true, cond.domain
    return None


def reduce(dim : FullDim, op : ReduceOpType, interval : tuple[SymbolicIndex, SymbolicIndex], child : NormalizedExpr):
    num, den = child.num, child.den

    varying_num, invariant_num = partition_by_dim_variance(dim, num)
    varying_den, invariant_den = partition_by_dim_variance(dim, den)

    # Hoist invariant factors out of the reduce. Once hoisted, any Repeat(dim, ...)
    # inside them is redundant since the enclosing shape no longer has dim.
    invariant_num_expr = strip_repeats_from_product(
        NormalizedProduct(const=num.const, factors=invariant_num), dim,
    )
    invariant_den_expr = strip_repeats_from_product(
        NormalizedProduct(const=1.0, factors=invariant_den), dim,
    )

    expr_to_reduce = make_expr(
        NormalizedProduct(const=1.0, factors=varying_num),
        NormalizedProduct(const=1.0, factors=varying_den),
    )

    # Generic tagged-body fold: if the reduce's body is a single tagged
    # tensor `Cond(D, body, identity)`, strip the tag and intersect D into
    # the reduce's domain. Subsumes the old "body * mask" extraction.
    base_domain = interval_domain(_reduce_index(dim), [interval])
    extracted = _extract_tagged_body(expr_to_reduce, op)
    if extracted is not None:
        expr_to_reduce, tag_domain = extracted
        final_domain = and_domains(base_domain, tag_domain)
        # The fold may have surfaced new invariant factors (e.g., when
        # `div` pushed through the tag, the rhs's now-unwrapped body
        # ends up inside the if_true branch — and any of those factors
        # that don't depend on `dim` should be hoisted just like the
        # pre-fold partition did at the top of `reduce`).
        post_varying_num, post_invariant_num = partition_by_dim_variance(
            dim, expr_to_reduce.num,
        )
        post_varying_den, post_invariant_den = partition_by_dim_variance(
            dim, expr_to_reduce.den,
        )
        post_inv_num_expr = strip_repeats_from_product(
            NormalizedProduct(const=expr_to_reduce.num.const, factors=post_invariant_num),
            dim,
        )
        post_inv_den_expr = strip_repeats_from_product(
            NormalizedProduct(const=1.0, factors=post_invariant_den),
            dim,
        )
        invariant_num_expr = mul(invariant_num_expr, post_inv_num_expr)
        invariant_den_expr = mul(invariant_den_expr, post_inv_den_expr)
        expr_to_reduce = make_expr(
            NormalizedProduct(const=1.0, factors=post_varying_num),
            NormalizedProduct(const=1.0, factors=post_varying_den),
        )
    else:
        final_domain = base_domain

    reduction = make_reduce(
        dim=dim_full_dim(dim),
        op=op,
        intervals=final_domain,
        child=expr_to_reduce,
    )
    reduction_expr = NormalizedExpr.of(reduction)
    return div(mul(invariant_num_expr, reduction_expr), invariant_den_expr)


def _normalize_tag(tag : TagTree) -> NormalizedTagTree:
    """
    Recursively normalize a `TagTree`. Leaves are `ExprType`s, so they
    normalize to `NormalizedExpr`s; branches recurse. `Cond(D, t, t)`
    with identical branches collapses to `t`.
    """
    if isinstance(tag, TagCond):
        nt = _normalize_tag(tag.if_true)
        nf = _normalize_tag(tag.if_false)
        if nt == nf:
            return nt
        return NormalizedTagCond(tag.domain, nt, nf)
    # tag is an ExprType leaf.
    return normalize(tag)


@functools.cache
def normalize(expr : ExprType) -> NormalizedExpr:
    match expr:
        case Constant(x):
            const = NormalizedProduct(const=x)
            return NormalizedExpr.of(const)
        case Tensor(dims, tag):
            norm_tag = _normalize_tag(tag) if tag is not None else None
            t = NormalizedTensor(dims=frozenset(dims), tag=norm_tag)
            return NormalizedExpr.of(t)
        case UnaryOp(op, child):
            return unary_op(op, normalize(child))
        case BinaryOp(op, lhs, rhs):
            return binary_op(op, normalize(lhs), normalize(rhs))
        case Repeat(dim, child):
            # Repeat is special - don't pre-normalize!
            return repeat(frozenset([dim]), child)
        case Reduce(op, dim, child):
            normalized_child = normalize(child)
            return reduce(
                dim=dim_full_dim(dim),
                op=op,
                interval=(dim_start(dim), dim_end(dim)),
                child=normalized_child,
            )
        case ParametricReduce(loop_var, lo, hi, op, body):
            return make_parametric_reduce(loop_var, lo, hi, op, normalize(body))


def verify_exprs_equivalent(x : ExprType, y : ExprType) -> bool:
    return normalize(x) == normalize(y)

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
