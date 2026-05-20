import builtins
import functools
import math
from dataclasses import dataclass, field

from .type import *
from .indexing import (
    Domain, AffineConstraint, interval_domain, ge, lt, and_domains,
    simplify_domain, active_loop_vars, _active_loop_scopes,
    runtime_scalar_max, index_has_property, paired_index_for_offsets,
    resolve_symbolic_index,
)
from .frozen_counter import FrozenCounter


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


def make_tag_cond(
    domain : Domain, if_true : "NormalizedTagTree", if_false : "NormalizedTagTree",
) -> "NormalizedTagTree":
    """
    Canonical constructor for `NormalizedTagCond`. Two collapse rules:

    1. **Same-predicate**: inside the `if_true` branch, `domain` is true,
       so any nested `Cond(domain, A, B)` resolves to `A`; symmetrically
       in `if_false`. This collapses arbitrarily deep nestings of the
       same predicate to a single `Cond`, giving `mask^k = mask`
       (boolean idempotence under multiplication).
    2. **Trivial branches**: when `if_true == if_false`, the value is
       independent of `domain` — return the leaf directly. Combined
       with constant propagation through `max`/`+`/`*`/etc., this
       collapses e.g. `max(Cond(P, 0, -inf), 0)` → `Cond(P, 0, 0)` →
       `0`. The return type widens to `NormalizedTagTree` so the leaf
       can flow back out of the construction site.
    """
    if isinstance(if_true, NormalizedTagCond) and if_true.domain == domain:
        if_true = if_true.if_true
    if isinstance(if_false, NormalizedTagCond) and if_false.domain == domain:
        if_false = if_false.if_false
    if if_true == if_false:
        return if_true
    return NormalizedTagCond(domain, if_true, if_false)


# NormalizedTagTree = NormalizedExpr | NormalizedTagCond
# Defined after NormalizedExpr is declared.


@dataclass(frozen=True)
class NormalizedTensor:
    dims : frozenset[FullDim]
    tag : "NormalizedTagTree | None" = None
    name : str = ""

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash((self.dims, self.tag, self.name))
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


@dataclass(frozen=True)
class NormalizedGather:
    """
    `source.gather(dim_in_source, idx)` in normalized form. Treated
    opaquely: two `NormalizedGather`s with equal `source`,
    `dim_in_source`, and `idx` are structurally equal; the verifier
    doesn't introspect `idx`'s values. Acts like a `NormalizedTensor`
    in factor lists — a leaf for the purposes of normalization and
    dim-variance partitioning.
    """
    source : "NormalizedExpr"
    dim_in_source : FullDim
    idx : "NormalizedExpr"

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash((self.source, self.dim_in_source, self.idx))
        object.__setattr__(self, '_h', h)
        return h


@dataclass(frozen=True)
class NormalizedScatter:
    """
    `source.scatter(dim_in_dest, idx)` in normalized form. Dual of
    `NormalizedGather`. Same opaque-leaf treatment.
    """
    source : "NormalizedExpr"
    dim_in_dest : FullDim
    idx : "NormalizedExpr"

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash((self.source, self.dim_in_dest, self.idx))
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


def _substitute_in_symint(
    atom : "LoopVariable",
    loop_var : "LoopVariable",
    value : SymbolicIndex,
) -> "LoopVariable":
    """
    If `atom.source` contains `loop_var` inside its `position` field,
    return a new `SymbolicInt` with the position recursively
    substituted. Otherwise return `atom` unchanged. This is what lets
    a `tensor_element("offsets", g)` substitution flow through
    `_substitute_loop_var`: without this, the `source` field is
    opaque to substitution and `te(offsets, g)` stays untouched even
    when we substitute `g → k` outside of it.
    """
    from .indexing import SymbolicInt
    if atom.source is None:
        return atom
    tensor_name, position = atom.source
    new_position = _substitute_loop_var(position, loop_var, value)
    if new_position == position:
        return atom
    return SymbolicInt(name=atom.name, source=(tensor_name, new_position))


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
            # If `v` is a `tensor_element` atom whose `position` field
            # references `loop_var`, push the substitution into the
            # source. Otherwise `v` flows through unchanged.
            new_v = _substitute_in_symint(v, loop_var, value)
            result = result + c * new_v
    return result


def substitute_loop_var_in_et(
    et : ExprType,
    loop_var : LoopVariable,
    value : SymbolicIndex,
) -> ExprType:
    """
    Walk an `ExprType` AST and substitute every occurrence of
    `loop_var` with `value`. Affects:

      - `Sliced` dim bounds (in `Tensor.dims`, `Repeat.dim`,
        `Reduce.dim`, and nested).
      - `TagCond` domain constraints (recursively).
      - `ParametricReduce` `lo`/`hi` (and inner body, but not when
        the parametric loop variable matches `loop_var`'s name —
        that's a binder shadow).

    Used by `fori_loop(invariant=...)` to materialize the loop's
    return value as `invariant[k=upper]` at the AST level so
    downstream `tjax` ops can keep building.
    """
    def sub_dim(d):
        match d:
            case FullDim():
                return d
            case Sliced(child, start, end):
                return Sliced(
                    sub_dim(child),
                    _substitute_loop_var(start, loop_var, value),
                    _substitute_loop_var(end, loop_var, value),
                )
        return d

    def sub_domain(domain):
        new_disjuncts = frozenset(
            frozenset(
                AffineConstraint(
                    _substitute_loop_var(c.expr, loop_var, value)
                )
                for c in conj
            )
            for conj in domain.disjuncts
        )
        new_vars = frozenset(v for v in domain.variables if v != loop_var)
        # Add any free vars that came in via `value`.
        new_vars |= frozenset(
            v for c in to_affine(value).terms for (v, _) in [c]
        )
        return Domain(new_vars, new_disjuncts)

    def sub_tag(tag):
        if isinstance(tag, TagCond):
            return TagCond(
                domain=sub_domain(tag.domain),
                if_true=sub_tag(tag.if_true),
                if_false=sub_tag(tag.if_false),
            )
        return sub_et(tag)

    def sub_et(et):
        match et:
            case Constant():
                return et
            case Tensor(dims=dims, tag=tag, name=name):
                new_tag = sub_tag(tag) if tag is not None else None
                return Tensor(
                    dims=tuple(sub_dim(d) for d in dims),
                    tag=new_tag,
                    name=name,
                )
            case UnaryOp(op, child):
                return UnaryOp(op, sub_et(child))
            case BinaryOp(op, lhs, rhs):
                return BinaryOp(op, sub_et(lhs), sub_et(rhs))
            case Repeat(dim, child):
                return Repeat(sub_dim(dim), sub_et(child))
            case Reduce(op, dim, child):
                return Reduce(op, sub_dim(dim), sub_et(child))
            case ParametricReduce(lv, lo, hi, op, body):
                if lv.name == loop_var.name:
                    return et  # shadowed by inner binder
                return ParametricReduce(
                    lv,
                    _substitute_loop_var(lo, loop_var, value),
                    _substitute_loop_var(hi, loop_var, value),
                    op,
                    sub_et(body),
                )
        return et

    return sub_et(et)


def _conjunction_is_infeasible(conj) -> bool:
    """
    True when this conjunction of `expr >= 0` constraints has no
    solutions over the integers — currently only the simple case
    where some single variable has a lower bound that exceeds its
    upper bound (after collapsing 1-D bounds). Catches the empty
    domain that arises when an invariant `sum[N where N < k]`
    is evaluated at `k=0`.
    """
    vars_in_conj = {v for c in conj for v, _ in c.expr.terms}
    for var in vars_in_conj:
        lo = None
        hi = None
        for c in conj:
            coeff = _loop_var_coeff(c.expr, var)
            if coeff == 1:
                residue = c.expr - to_affine(var)
                if not residue.terms:
                    cand = -residue.const
                    lo = cand if lo is None else max(lo, cand)
            elif coeff == -1:
                residue = c.expr + to_affine(var)
                if not residue.terms:
                    cand = residue.const
                    hi = cand if hi is None else min(hi, cand)
        if lo is not None and hi is not None and lo > hi:
            return True
    return False


def _domain_is_empty(domain : Domain) -> bool:
    """Every disjunct is infeasible → domain is empty (no iteration points)."""
    return all(_conjunction_is_infeasible(c) for c in domain.disjuncts)


def _substitute_lv_in_expr(
    expr : "NormalizedExpr",
    loop_var : LoopVariable,
    value : SymbolicIndex,
) -> "NormalizedExpr":
    """
    Replace every occurrence of `loop_var` inside `expr` with `value`,
    walking the full expression tree (Reduce intervals, nested children,
    etc.). Shadowing by an inner `ParametricReduce` binder is respected.
    Empty-domain reduces collapse to the op's identity (0 for sum,
    -inf for max).
    """
    return div(
        _substitute_lv_in_product(expr.num, loop_var, value),
        _substitute_lv_in_product(expr.den, loop_var, value),
    )


def _substitute_lv_in_product(
    product : "NormalizedProduct",
    loop_var : LoopVariable,
    value : SymbolicIndex,
) -> "NormalizedExpr":
    """
    Rebuild the product factor-by-factor via `mul` so that a factor
    whose substitution collapses to a Constant (e.g. an empty-domain
    sum-Reduce → 0) propagates through cleanly.
    """
    result = NormalizedExpr.of(NormalizedProduct(const=product.const))
    for factor, count in product.factors.items():
        sub = _substitute_lv_in_factor_to_expr(factor, loop_var, value)
        for _ in range(count):
            result = mul(result, sub)
    return result


def _substitute_lv_in_factor_to_expr(
    factor : "NormalizedFactor",
    loop_var : LoopVariable,
    value : SymbolicIndex,
) -> "NormalizedExpr":
    """
    Like `_substitute_lv_in_factor` but returns a `NormalizedExpr` so
    the result can be a non-factor (notably: identity-valued Constants
    when an empty Reduce collapses).
    """
    match factor:
        case NormalizedReduce(dim, op, domain, child):
            new_domain = simplify_domain(
                _substitute_lv_in_domain(domain, loop_var, value)
            )
            if _domain_is_empty(new_domain):
                identity = 0.0 if op == "sum" else float("-inf")
                return NormalizedExpr.of(NormalizedProduct(const=identity))
            new_child = _substitute_lv_in_expr(child, loop_var, value)
            # Inner reduce-of-identity collapses through: `sum_d 0 = 0`
            # and `max_d -inf = -inf`. Without this, a nested empty
            # reduce (whose substitution returned a const-identity
            # `NormalizedExpr`) leaves the outer reduce wrapping that
            # constant — structurally distinct from a bare `0` even
            # though they're algebraically equal.
            if (
                not new_child.num.factors
                and not new_child.den.factors
                and new_child.den.const == 1.0
            ):
                c = new_child.num.const
                if (op == "sum" and c == 0.0) or (
                    op == "max" and c == float("-inf")
                ):
                    return new_child
            return NormalizedExpr.of(
                NormalizedReduce(dim, op, new_domain, new_child)
            )
        case NormalizedParametricReduce(inner_var, lo, hi, op, body):
            # Route ParametricReduce substitution through
            # `make_parametric_reduce`, which collapses the empty range
            # `lo == hi` → identity. Without this, a `fori_loop`
            # invariant of the form
            # `ParametricReduce(g, 0, k, sum, ...)` doesn't normalize
            # to `0` at the base case `k = 0`.
            new_lo = _substitute_loop_var(lo, loop_var, value)
            new_hi = _substitute_loop_var(hi, loop_var, value)
            if inner_var == loop_var:
                return NormalizedExpr.of(NormalizedParametricReduce(
                    inner_var, new_lo, new_hi, op, body,
                ))
            new_body = _substitute_lv_in_expr(body, loop_var, value)
            return make_parametric_reduce(inner_var, new_lo, new_hi, op, new_body)
    # All other factor kinds: substitute via the factor-returning helper
    # (no identity-collapse needed) and wrap.
    return NormalizedExpr.of(_substitute_lv_in_factor(factor, loop_var, value))


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
        case NormalizedGather(source, dim_in_source, idx):
            return NormalizedGather(
                source=_substitute_lv_in_expr(source, loop_var, value),
                dim_in_source=dim_in_source,
                idx=_substitute_lv_in_expr(idx, loop_var, value),
            )
        case NormalizedScatter(source, dim_in_dest, idx):
            return NormalizedScatter(
                source=_substitute_lv_in_expr(source, loop_var, value),
                dim_in_dest=dim_in_dest,
                idx=_substitute_lv_in_expr(idx, loop_var, value),
            )

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
    interval_vars = frozenset(
        v for (s, e) in merged_intervals
        for x in (s, e)
        for (v, _) in to_affine(x).terms
    )
    disjuncts : set = set()
    for s, e in merged_intervals:
        conj = set(extras)
        conj.add(ge(index, s))
        conj.add(lt(index, e))
        disjuncts.add(frozenset(conj))
    domain = Domain(
        frozenset({index}) | extra_vars | interval_vars, frozenset(disjuncts),
    )
    domain = simplify_domain(domain)
    return make_reduce(dim, op, domain, child)


def _max_in_loop_scope(expr : SymbolicIndex) -> int | None:
    """
    Upper bound on `expr` using the currently-active `LoopScope` iteration
    ranges and the natural ranges of any free dim variables (a
    `LoopVariable` named after a registered `FullDim` ranges over
    `[0, dim.size)`). Returns `None` when no range is available for some
    variable — e.g. a free `LoopVariable` that's neither a loop scope nor
    a dim — or when an endpoint isn't concretely an integer.
    """
    aff = to_affine(expr)
    total = aff.const
    scopes_by_var = {s.var : s for s in _active_loop_scopes}
    for v, c in aff.terms:
        scope = scopes_by_var.get(v)
        if scope is not None:
            lo_int = as_int(scope.lo)
            hi_int = as_int(scope.hi)
        else:
            registered = g_dim_registry.get(v.name)
            if registered is not None:
                lo_int, hi_int = 0, registered.size
            else:
                rs_max = runtime_scalar_max(v.name)
                if rs_max is None:
                    return None
                lo_int, hi_int = 0, rs_max
        if lo_int is None or hi_int is None:
            return None
        # max at `hi - 1` if c > 0, else at `lo`.
        total += c * ((hi_int - 1) if c > 0 else lo_int)
    return total


def simplify_under_active_loop_scope(expr : "NormalizedExpr") -> "NormalizedExpr":
    """
    Re-walk `expr` and drop redundant natural bounds from every
    `NormalizedReduce`'s domain, using the currently-active loop scopes
    to prove subsumption. Used by `_fori_loop_with_invariant` to compare
    a body's output against `invariant[k+1]` under the loop scope's
    bounds; without this step the unmerged invariant form keeps its
    natural `n < dim.size` constraint, while the merged body output (its
    interval already supplies a tighter loop-var bound) doesn't, so they
    canonicalize differently.
    """
    def walk_expr(e : NormalizedExpr) -> NormalizedExpr:
        new_num = walk_product(e.num)
        new_den = walk_product(e.den)
        if new_num is e.num and new_den is e.den:
            return e
        return NormalizedExpr(num=new_num, den=new_den)

    def walk_product(p : NormalizedProduct) -> NormalizedProduct:
        new_factors_items = []
        changed = False
        for f, count in p.factors.items():
            new_f = walk_factor(f)
            if new_f is not f:
                changed = True
            new_factors_items.append((new_f, count))
        if not changed:
            return p
        new_factors = FrozenCounter.from_dict(dict(new_factors_items))
        return NormalizedProduct(const=p.const, factors=new_factors)

    def walk_factor(f):
        match f:
            case NormalizedReduce(dim, op, domain, child):
                new_child = walk_expr(child)
                new_domain = _drop_redundant_natural_bounds(domain, dim)
                if new_domain == domain and new_child is child:
                    return f
                return NormalizedReduce(dim, op, new_domain, new_child)
            case NormalizedSum(terms):
                new_terms = [walk_product(t) for t in terms]
                if all(t is o for t, o in zip(new_terms, terms)):
                    return f
                return NormalizedSum(frozenset(new_terms))
            case NormalizedMax(children):
                new_children = [walk_expr(c) for c in children]
                if all(c is o for c, o in zip(new_children, children)):
                    return f
                return NormalizedMax(frozenset(new_children))
            case NormalizedExp(child):
                new_child = walk_expr(child)
                return f if new_child is child else NormalizedExp(new_child)
            case NormalizedUnaryOp(op, child):
                new_child = walk_expr(child)
                return f if new_child is child else NormalizedUnaryOp(op, new_child)
            case NormalizedRepeat(dim, child):
                new_child = walk_expr(child)
                return f if new_child is child else NormalizedRepeat(dim, new_child)
            case NormalizedParametricReduce(loop_var, lo, hi, op, body):
                new_body = walk_expr(body)
                return f if new_body is body else NormalizedParametricReduce(
                    loop_var, lo, hi, op, new_body,
                )
            case NormalizedGather(source, dim_in_source, idx):
                new_source = walk_expr(source)
                new_idx = walk_expr(idx)
                if new_source is source and new_idx is idx:
                    return f
                return NormalizedGather(new_source, dim_in_source, new_idx)
            case NormalizedScatter(source, dim_in_dest, idx):
                new_source = walk_expr(source)
                new_idx = walk_expr(idx)
                if new_source is source and new_idx is idx:
                    return f
                return NormalizedScatter(new_source, dim_in_dest, new_idx)
            case _:
                return f

    return walk_expr(expr)


def _drop_redundant_natural_bounds(
    domain : Domain, dim : FullDim,
) -> Domain:
    """
    Drop constant bounds on the reduce index that are subsumed by a
    symbolic bound on the same side, using active `LoopScope`s and free
    dim variables' natural ranges to bound the symbolic bound's
    maximum. Constant `idx < C` is dropped when some symbolic `idx < X`
    has `max(X) ≤ C`; symmetrically for lower bounds. Generalizes the
    natural-bound drop: lets a loop-var iteration bound `n < BN*k`
    subsume the `n < dim.size` natural that mask-form parsing pulls in,
    and lets a cross-variable causal bound `n <= q` subsume both the
    natural `n < nctx_size` and a post-substitution `n < BN*K` when
    `qctx_size ≤ min(BN*K, nctx_size)` — which is what makes
    "skip-tail" rolled causal flash match its softmax-attention spec.
    """
    index = _reduce_index(dim)
    new_disjuncts : set = set()
    for conj in domain.disjuncts:
        constraints = list(conj)
        constant_uppers : list[tuple[int, AffineConstraint]] = []
        constant_lowers : list[tuple[int, AffineConstraint]] = []
        symbolic_uppers : list[SymbolicIndex] = []
        symbolic_lowers : list[SymbolicIndex] = []
        for c in constraints:
            coeff = _loop_var_coeff(c.expr, index)
            if coeff == 1:
                residue = c.expr - to_affine(index)
                if not residue.terms:
                    constant_lowers.append((-residue.const, c))
                else:
                    symbolic_lowers.append(-residue)
            elif coeff == -1:
                residue = c.expr + to_affine(index)
                if not residue.terms:
                    constant_uppers.append((residue.const + 1, c))
                else:
                    symbolic_uppers.append(residue + 1)
        for const_val, const_c in constant_uppers:
            for sym in symbolic_uppers:
                mx = _max_in_loop_scope(sym)
                if mx is not None and mx <= const_val:
                    if const_c in constraints:
                        constraints.remove(const_c)
                    break
        for const_val, const_c in constant_lowers:
            for sym in symbolic_lowers:
                # min(sym) = -max(-sym).
                neg_max = _max_in_loop_scope(-sym)
                if neg_max is not None and -neg_max >= const_val:
                    if const_c in constraints:
                        constraints.remove(const_c)
                    break
        new_disjuncts.add(frozenset(constraints))
    return Domain(domain.variables, frozenset(new_disjuncts))


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

    Interval-bound priority is `loop-var-symbolic > constant > cross-var-
    symbolic`. Cross-variable predicates (e.g. `n < q + 1` in causal
    flash) stay in extras so tiles merge across concrete tile boundaries.
    Loop-variable iteration bounds (e.g. `n < BN*k` from a `where`-clause
    that references the active loop var) outrank constants so the
    accumulating invariant and its current tile share extras and merge
    via interval-union. A "natural" constant bound (`0` for lower, `dim.size`
    for upper) is dropped when a loop-var-symbolic takes its direction —
    it's implied by the reduce's range and would otherwise prevent the
    merge.
    """
    if len(domain.disjuncts) != 1:
        return None
    conj = next(iter(domain.disjuncts))
    index = _reduce_index(dim)
    loop_vars = active_loop_vars()

    def _is_loop_var_bound(expr : SymbolicIndex) -> bool:
        return any(v in loop_vars for v, _ in to_affine(expr).terms)

    constant_lowers : list[SymbolicIndex] = []
    loop_lowers : list[SymbolicIndex] = []
    crossvar_lowers : list[SymbolicIndex] = []
    constant_uppers : list[SymbolicIndex] = []
    loop_uppers : list[SymbolicIndex] = []
    crossvar_uppers : list[SymbolicIndex] = []
    extras : list[AffineConstraint] = []

    for c in conj:
        coeff = _loop_var_coeff(c.expr, index)
        if coeff == 1:
            residue = c.expr - to_affine(index)
            bound = -residue
            if not residue.terms:
                constant_lowers.append(bound)
            elif _is_loop_var_bound(bound):
                loop_lowers.append(bound)
            else:
                crossvar_lowers.append(bound)
        elif coeff == -1:
            residue = c.expr + to_affine(index)
            bound = residue + 1
            if not residue.terms:
                constant_uppers.append(bound)
            elif _is_loop_var_bound(bound):
                loop_uppers.append(bound)
            else:
                crossvar_uppers.append(bound)
        else:
            extras.append(c)

    natural_lower = to_affine(0)
    natural_upper = to_affine(dim.size) if isinstance(dim, FullDim) else None

    def _pick_lower():
        if loop_lowers:
            chosen = loop_lowers[0]
            leftovers : list[SymbolicIndex] = []
            for b in loop_lowers[1:] + constant_lowers + crossvar_lowers:
                if b == natural_lower:
                    continue
                leftovers.append(b)
            return chosen, leftovers
        if constant_lowers:
            return constant_lowers[0], constant_lowers[1:] + crossvar_lowers
        if crossvar_lowers:
            return crossvar_lowers[0], crossvar_lowers[1:]
        return None, []

    def _pick_upper():
        if loop_uppers:
            chosen = loop_uppers[0]
            leftovers : list[SymbolicIndex] = []
            for b in loop_uppers[1:] + constant_uppers + crossvar_uppers:
                if natural_upper is not None and b == natural_upper:
                    continue
                leftovers.append(b)
            return chosen, leftovers
        if constant_uppers:
            return constant_uppers[0], constant_uppers[1:] + crossvar_uppers
        if crossvar_uppers:
            return crossvar_uppers[0], crossvar_uppers[1:]
        return None, []

    start, leftover_lowers = _pick_lower()
    end, leftover_uppers = _pick_upper()
    for b in leftover_lowers:
        extras.append(ge(index, b))
    for b in leftover_uppers:
        extras.append(lt(index, b))

    if start is None or end is None:
        return None
    interval = (_concretize(start), _concretize(end))
    return frozenset({interval}), frozenset(extras)


def _pull_common_outer_reduce(
    terms : list["NormalizedProduct"],
) -> list["NormalizedProduct"]:
    """
    Sum-linearity for reduces: `Reduce(d, A) + Reduce(d, B) = Reduce(d, A + B)`.
    When two terms in a sum are bare sum-Reduces over the same dim and
    domain but with different children, pull the outer Reduce out so
    the children can subsequently tile-merge inside it via
    `_merge_sum_reduces`. This unlocks the multi-dim tile-walk pattern
    `sum_D(sum_n_tile_a(X)) + sum_D(sum_n_tile_b(X))` → `sum_D(sum_n_full(X))`,
    which doesn't reach `_merge_sum_reduces` directly because the outer
    `D`-reduce hides the inner tile structure from grouping by child.
    Only fires for `const == 1.0`, single-factor terms — multi-factor
    products would require shared "other factors" we don't try to
    intersect here.
    """
    groups : dict[tuple, list["NormalizedExpr"]] = {}
    others : list[NormalizedProduct] = []
    for t in terms:
        if t.const != 1.0 or len(t.factors) != 1:
            others.append(t)
            continue
        single = _as_single_factor(
            NormalizedProduct(const=1.0, factors=t.factors),
        )
        if isinstance(single, NormalizedReduce) and single.op == "sum":
            key = (single.dim, single.op, single.domain)
            groups.setdefault(key, []).append(single.child)
            continue
        others.append(t)

    pulled : list[NormalizedProduct] = []
    for (dim, op, domain), children in groups.items():
        if len(children) == 1:
            r = NormalizedReduce(dim, op, domain, children[0])
        else:
            combined = children[0]
            for c in children[1:]:
                combined = add(combined, c)
            r = NormalizedReduce(dim, op, domain, combined)
        pulled.append(NormalizedProduct(
            const=1.0,
            factors=FrozenCounter.from_iterable([r]),
        ))
    return others + pulled


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


def _common_positive_scalar(children) -> "float | None":
    """
    If every child has form `c * single_factor` (`num.const = c`, exactly
    one `num.factor`, trivial denominator) with the same positive `c`,
    return `c`. Else `None`. Used by `make_max` to factor `Max(c*A, c*B)`
    → `c * Max(A, B)` when `c > 0`.
    """
    consts : set[float] = set()
    for c in children:
        if c.den.factors or c.den.const != 1.0:
            return None
        if c.num.const <= 0:
            return None
        if len(c.num.factors) != 1:
            return None
        consts.add(c.num.const)
    if len(consts) != 1:
        return None
    (const,) = consts
    return const


def make_max(children) -> "NormalizedExpr":
    """Build a canonical NormalizedMax. Flattens nested maxes, drops -inf terms
    (identity for max), distributes max through tagged-tensor branches,
    merges max-Reduce children that share dim and child by unioning their
    intervals, absorbs boundary terms into ParametricMax, dedupes, and
    collapses singletons.
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

    # Distribute max through any tagged-tensor child: `max(Cond(P, a, b), x,
    # …) = Cond(P, max(a, x, …), max(b, x, …))`. The recursive `make_max`
    # call propagates constant values from x into both branches of P,
    # giving `max(bias_mask, 0) → Cond(P, max(0, 0), max(-inf, 0)) →
    # Cond(P, 0, 0) → 0` and `max(mult_mask, 0) → mult_mask`. Mirrors the
    # `+/-/*//` propagation done by `_distribute_binop_through_tag`.
    for i, c in enumerate(flat):
        tagged = _single_tagged_tensor(c)
        if tagged is not None and isinstance(tagged.tag, NormalizedTagCond):
            others = flat[:i] + flat[i + 1:]
            return _push_through_tag(
                lambda leaf: make_max([leaf, *others]),
                tagged,
            )

    # Fold pure-constant children into a single constant via builtin
    # `max`. Required for mask-propagation to reach its full simplification
    # — after pushing the surrounding `max(..., c)` into a `Cond`'s
    # branches, each branch contains `max(leaf, c)` of two constants
    # which needs constant folding to collapse to a single constant.
    const_children = [c for c in flat if _is_pure_const(c)]
    if len(const_children) > 1:
        max_const = const_children[0].num.const
        for c in const_children[1:]:
            if c.num.const > max_const:
                max_const = c.num.const
        flat = [c for c in flat if not _is_pure_const(c)]
        flat.append(
            NormalizedExpr.of(NormalizedProduct(const=max_const))
        )

    # Pull a common positive scalar out: `Max(c*A, c*B, …) = c*Max(A, B, …)`
    # when `c > 0`. Lets per-tile reduces with a shared scaling
    # (e.g. `qk / sqrt(d)` per tile) tile-merge — the inner `Max(A, B, …)`
    # then has pure single-factor children that the per-Reduce grouping
    # below can pick up.
    common = _common_positive_scalar(flat)
    if common is not None and common != 1.0:
        scaled = [
            NormalizedExpr.of(NormalizedProduct(
                const=c.num.const / common,
                factors=c.num.factors,
            ))
            for c in flat
        ]
        inner = make_max(scaled)
        return mul(
            NormalizedExpr.of(NormalizedProduct(const=common)),
            inner,
        )

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

    flat = _pull_common_outer_reduce(flat)
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

NormalizedFactor = NormalizedTensor | NormalizedExp | NormalizedUnaryOp | NormalizedSum | NormalizedMax | NormalizedRepeat | NormalizedReduce | NormalizedParametricReduce | NormalizedGather | NormalizedScatter
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
            return make_tag_cond(
                domain=tag.domain,
                if_true=apply(tag.if_true),
                if_false=apply(tag.if_false),
            )
        return transform(tag)
    new_tag = apply(tensor.tag)
    # If the tag collapsed to a pure-constant leaf (no factors), the
    # tensor's value is independent of position — return the constant
    # directly instead of a degenerate `Tensor(tag=Constant)`. This is
    # what makes `max(bias_mask, 0) → Cond(P, 0, 0) → 0` actually land
    # at `0` after the `if_true == if_false` collapse in `make_tag_cond`.
    if isinstance(new_tag, NormalizedExpr) and _is_pure_const(new_tag):
        return new_tag
    return NormalizedExpr.of(
        NormalizedTensor(dims=tensor.dims, tag=new_tag, name=tensor.name)
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
    # Both tagged. Same-predicate fast path: when both tags are a
    # `Cond(P, …)` with the *same* domain, the unreachable cross-branch
    # cells (P-true ⊗ P-false) never fire. Combine branch-by-branch
    # into a single `Cond(P, op(t_l, t_r), op(f_l, f_r))`. This is what
    # makes `mask^k = mask` (boolean idempotence): without it, repeated
    # mults of the same mask produce arbitrarily deep nested same-P
    # `Cond` trees that don't structurally compare equal to the single
    # `Cond` form.
    if (
        isinstance(lhs_tag.tag, NormalizedTagCond)
        and isinstance(rhs_tag.tag, NormalizedTagCond)
        and lhs_tag.tag.domain == rhs_tag.tag.domain
        and lhs_tag.dims == rhs_tag.dims
    ):
        l_tag, r_tag = lhs_tag.tag, rhs_tag.tag
        def _combine_branches(l_leaf, r_leaf):
            return op_fn(_leaf_to_expr(l_leaf), _leaf_to_expr(r_leaf))
        new_tag = make_tag_cond(
            domain=l_tag.domain,
            if_true=_combine_branches(l_tag.if_true, r_tag.if_true),
            if_false=_combine_branches(l_tag.if_false, r_tag.if_false),
        )
        # If both branches collapsed to the same pure-constant leaf
        # (e.g. `mask - mask = Cond(P, 0, 0) → 0`), return that
        # constant directly instead of wrapping it back into a
        # degenerate `Tensor(tag=Const)`. Mirrors the same unwrap rule
        # `_push_through_tag` applies for single-tagged distribution.
        if isinstance(new_tag, NormalizedExpr) and _is_pure_const(new_tag):
            return new_tag
        return NormalizedExpr.of(NormalizedTensor(
            dims=lhs_tag.dims, tag=new_tag, name=lhs_tag.name,
        ))
    # General nest (outer = lhs's tag, inner = rhs's tag).
    return _push_through_tag(
        lambda l_leaf: _push_through_tag(
            lambda r_leaf: op_fn(l_leaf, r_leaf), rhs_tag,
        ),
        lhs_tag,
    )


def _leaf_to_expr(leaf) -> "NormalizedExpr":
    """A `NormalizedTagTree` leaf is either a `NormalizedExpr` (the
    plain value at that branch) or a deeper `NormalizedTagCond`. The
    latter only appears when the tag itself is nested. For the
    same-predicate fast path in `_distribute_binop_through_tag`, we
    treat any nested `Cond` leaf by wrapping it back into a tagged
    tensor and lifting to a `NormalizedExpr`."""
    if isinstance(leaf, NormalizedTagCond):
        return NormalizedExpr.of(NormalizedTensor(
            dims=frozenset(), tag=leaf, name="_mask",
        ))
    return leaf


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
            NormalizedProduct(builtins.abs(term.const), term.factors)
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
            # Route through the same tag-distribution path as +/-/*//
            # before falling back to `make_max`. Without this, `max(Cond_P,
            # Cond_P)` would nest via `make_max`'s per-child propagation
            # — same-pred would never collapse because the inner Cond
            # ends up wrapped in a `NormalizedExpr`, not exposed at the
            # outer `make_tag_cond`'s branch.
            distributed = _distribute_binop_through_tag(
                lambda a, b: make_max([a, b]), nlhs, nrhs,
            )
            if distributed is not None:
                return distributed
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
                        return make_tag_cond(
                            domain=tag.domain,
                            if_true=apply(tag.if_true),
                            if_false=apply(tag.if_false),
                        )
                    return push(tag)
                return NormalizedExpr.of(
                    NormalizedTensor(
                        dims=tagged.dims | dims,
                        tag=apply(tagged.tag),
                        name=tagged.name,
                    )
                )
            return NormalizedExpr.of(NormalizedRepeat(dims, child_normalized))

def _tag_varies_with_dim(tag, dim : FullDim) -> bool:
    """A `NormalizedTagTree` "varies with `dim`" iff its domain
    constraints reference a variable named after `dim`, or either
    branch (recursively) varies with `dim`. Critical for the
    hoisting / Repeat-pushdown rules in `make_expr` /
    `_pull_common_outer_reduce`: a tagged tensor synthesized with
    `dims=frozenset()` (e.g. `_leaf_to_expr` wrapping a `Cond(P, …)`
    where `P` references a free dim variable) would otherwise be
    reported as dim-invariant and get incorrectly moved outside a
    Reduce / Repeat over that dim."""
    if isinstance(tag, NormalizedTagCond):
        if any(v.name == dim.name for v in tag.domain.variables):
            return True
        return (
            _tag_varies_with_dim(tag.if_true, dim)
            or _tag_varies_with_dim(tag.if_false, dim)
        )
    if isinstance(tag, NormalizedExpr):
        return varies_with_dim(tag, dim)
    return False


def varies_with_dim(e : NormalizedFactor | NormalizedProduct | NormalizedExpr, dim : FullDim) -> bool:
    """True iff the value of `e` changes as you move along `dim`.

    Repeat(dim, x) is *constant* along dim (broadcasts a single value) — returns False.
    Reduce(dim, x) eliminates dim, so the result is also constant along dim — returns False.
    """
    match e:
        case NormalizedTensor(dims=dims, tag=tag):
            if dim in dims:
                return True
            if tag is not None:
                return _tag_varies_with_dim(tag, dim)
            return False
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
        case NormalizedGather(source, dim_in_source, idx):
            # The source's `dim_in_source` is *replaced* in the output —
            # the gather no longer varies along it (varies along `idx`'s
            # dims instead). Any other dim of source flows through;
            # `idx`'s own dims contribute.
            if dim == dim_in_source:
                return False
            return varies_with_dim(source, dim) or varies_with_dim(idx, dim)
        case NormalizedScatter(source, dim_in_dest, idx):
            # Dual of gather: source's idx-dim is replaced by `dim_in_dest`
            # in the output. Output varies along `dim_in_dest` (a fresh
            # dim from the scatter's perspective) and along source's
            # other dims; idx's dim is consumed.
            if dim == dim_in_dest:
                return True
            return varies_with_dim(source, dim) or varies_with_dim(idx, dim)
        case NormalizedParametricReduce(_, _, _, _, body):
            return varies_with_dim(body, dim)
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
        case NormalizedGather(source, dim_in_source, idx):
            return NormalizedExpr.of(NormalizedGather(
                source=strip_repeats_from_expr(source, dim),
                dim_in_source=dim_in_source,
                idx=strip_repeats_from_expr(idx, dim),
            ))
        case NormalizedScatter(source, dim_in_dest, idx):
            return NormalizedExpr.of(NormalizedScatter(
                source=strip_repeats_from_expr(source, dim),
                dim_in_dest=dim_in_dest,
                idx=strip_repeats_from_expr(idx, dim),
            ))
        case NormalizedParametricReduce(_, _, _, _, _):
            return NormalizedExpr.of(factor)

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
    # Block-constant rewrite: if this reduce is over a block-bounded
    # slice `[offsets[g], offsets[g+1])` and an index paired with
    # `offsets` appears as a gather idx in the body, the gather is
    # `g`-everywhere on the block — replace it with `W[g]` (a
    # singleton slice of the gather's source). This is what unlocks
    # fused MoE: `gather(W, eid_sorted)` inside a per-expert block
    # collapses to `W[g]`, so the kernel's per-expert `W[g] @ X_block`
    # form matches its per-token-gather spec.
    block = _detect_block_interval(interval)
    if block is not None:
        offsets_name, g_pos = block
        paired = paired_index_for_offsets(offsets_name)
        if paired is not None:
            # The substitution may produce a body in which the gather
            # has been replaced by a `NormalizedReduce` over the
            # gather's source dim — the gather's contribution to the
            # body's shape is gone, and the new reduce is independent
            # of `dim`, so it'll hoist out cleanly below.
            child = _substitute_gather_with_block_constant(
                child, paired, dim, g_pos,
            )

    num, den = child.num, child.den

    varying_num, invariant_num = partition_by_dim_variance(dim, num)
    varying_den, invariant_den = partition_by_dim_variance(dim, den)

    # Gather-through-reduce: if every varying factor is a Gather with the
    # SAME `(dim_in_source, idx)` and the reduce dim doesn't intersect
    # the gather's output dim (`varies_with_dim(idx, dim)` is False),
    # factor the common gather out:
    #   Reduce(d, op, gather(A, π) · gather(B, π)) → gather(Reduce(d, op, A · B), π)
    # This is what makes sort-based MoE collapse: after `argsort=π`,
    # both `X_sorted` and `W_per_sorted` are gathers with idx `π`, so a
    # `Reduce(d_in, sum, ...)` over them lifts `π` out, leaving the
    # per-token spec inside, which the surrounding `scatter(_, π)`
    # then unwraps via the permutation round-trip.
    factored = _try_factor_common_gather(varying_num, dim, op)
    if factored is not None and not varying_den and num.const == 1.0:
        gather_dim, gather_idx, inner_body = factored
        invariant_num_expr = strip_repeats_from_product(
            NormalizedProduct(const=num.const, factors=invariant_num), dim,
        )
        invariant_den_expr = strip_repeats_from_product(
            NormalizedProduct(const=1.0, factors=invariant_den), dim,
        )
        inner = reduce(dim, op, interval, inner_body)
        gathered = NormalizedExpr.of(NormalizedGather(
            source=inner,
            dim_in_source=gather_dim,
            idx=gather_idx,
        ))
        return div(mul(invariant_num_expr, gathered), invariant_den_expr)

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

    if isinstance(final_domain, Domain) and _domain_is_empty(final_domain):
        # Empty iteration range: sum-reduce → 0 (annihilates), max-reduce
        # → -inf (identity). Skip the wrap so a `sum[d in ∅]` doesn't
        # survive as a structurally non-zero NormalizedReduce.
        identity = 0.0 if op == "sum" else float("-inf")
        return NormalizedExpr.of(NormalizedProduct(const=identity))
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
        return make_tag_cond(tag.domain, nt, nf)
    # tag is an ExprType leaf.
    return normalize(tag)


def _as_tensor_element(idx : SymbolicIndex) -> "LoopVariable | None":
    """
    If `idx` is exactly a single `SymbolicInt` with a `source` field
    (i.e. created via `tensor_element`), return that atom. Else `None`.
    Used by the block-constant rewrite to detect when a slice bound
    came from `offsets[g]`-style tensor indexing.
    """
    aff = to_affine(idx)
    if aff.const != 0 or len(aff.terms) != 1:
        return None
    (var, coeff), = aff.terms
    if coeff != 1 or var.source is None:
        return None
    return var


def _detect_block_interval(
    interval : tuple[SymbolicIndex, SymbolicIndex],
) -> "tuple[str, SymbolicIndex] | None":
    """
    If `interval` is `(tensor_element(T, g), tensor_element(T, g+1))`
    for some tensor `T` and position `g`, return `(T, g)`. Else `None`.
    The position `g` is returned as a `SymbolicIndex` (may be a free
    `SymbolicInt`, an `AffineExpr`, etc.).
    """
    lo, hi = interval
    lo_te = _as_tensor_element(lo)
    hi_te = _as_tensor_element(hi)
    if lo_te is None or hi_te is None:
        return None
    lo_tensor, lo_pos = lo_te.source
    hi_tensor, hi_pos = hi_te.source
    if lo_tensor != hi_tensor:
        return None
    # hi's position must be lo's position + 1.
    if to_affine(hi_pos) != to_affine(lo_pos) + 1:
        return None
    return lo_tensor, lo_pos


def _substitute_gather_with_block_constant(
    expr : "NormalizedExpr",
    idx_name : str,
    n_groups_dim : FullDim,
    g : SymbolicIndex,
) -> "NormalizedExpr":
    """
    Walk `expr` and replace each `NormalizedGather` whose idx is the
    bare tensor named `idx_name` with the singleton slice of the
    gather's source at position `g`:
      `gather(W, n_groups_dim, idx_name) → Reduce(n_groups_dim, sum,
      [g, g+1), W)`.
    The shape changes — the gather's idx-shape dim is replaced by the
    reduce-to-singleton — but inside the block-bounded slice the
    values agree (idx==g on the block, so gather picks W[g]
    everywhere). The shape change is absorbed by the surrounding
    block-bounded Reduce: the now-`n_tokens`-invariant `W[g]` hoists
    out cleanly. The gather's `dim_in_source` is what gets reduced;
    we use the gather's recorded `dim_in_source` from its
    NormalizedGather node, not the caller-supplied `n_groups_dim`,
    in case different gathers in the same expression use different
    source dims.
    """
    def walk_factor(f):
        if isinstance(f, NormalizedGather):
            idx_atom = None
            if f.idx.num.const == 1.0 and not f.idx.den.factors and f.idx.den.const == 1.0:
                inner = _as_single_factor(f.idx.num)
                if isinstance(inner, NormalizedTensor) and inner.tag is None:
                    idx_atom = inner
            if idx_atom is not None and idx_atom.name == idx_name:
                # Build the singleton slice: Reduce(dim_in_source, sum,
                # [g, g+1), source). After this single-element reduce
                # the result has the same shape as W minus dim_in_source —
                # the per-token output dim of the gather is gone, which
                # is fine because the surrounding block reduces over it.
                index = _reduce_index(f.dim_in_source)
                domain = interval_domain(
                    index, [(g, to_affine(g) + 1)],
                )
                return NormalizedReduce(
                    f.dim_in_source, "sum",
                    simplify_domain(domain),
                    f.source,
                )
        return f
    return _walk_expr_replacing_factors(expr, walk_factor)


def _walk_expr_replacing_factors(
    expr : "NormalizedExpr",
    walk : "Callable[[NormalizedFactor], NormalizedFactor]",
) -> "NormalizedExpr":
    """
    Rebuild `expr` with `walk` applied to every factor in num/den,
    recursing into nested `NormalizedExpr` children (Reduce/Repeat/
    Exp/UnaryOp/Sum/Max/etc.). `walk` returns `f` unchanged to skip,
    or a new factor to substitute. Recursion happens after a no-op
    `walk` — so a substituting walker that returns a new factor for
    `f` doesn't also rewrite inside the new factor.
    """
    def walk_factor(f):
        new_f = walk(f)
        if new_f is not f:
            return new_f
        match f:
            case NormalizedReduce(dim, op, domain, child):
                new_child = walk_expr(child)
                if new_child is child:
                    return f
                return NormalizedReduce(dim, op, domain, new_child)
            case NormalizedRepeat(dims, child):
                new_child = walk_expr(child)
                if new_child is child:
                    return f
                return NormalizedRepeat(dims, new_child)
            case NormalizedExp(child):
                new_child = walk_expr(child)
                if new_child is child:
                    return f
                return NormalizedExp(new_child)
            case NormalizedUnaryOp(op, child):
                new_child = walk_expr(child)
                if new_child is child:
                    return f
                return NormalizedUnaryOp(op, new_child)
            case NormalizedSum(children):
                new_children = [walk_product(c) for c in children]
                if all(nc is c for nc, c in zip(new_children, children)):
                    return f
                return NormalizedSum(frozenset(new_children))
            case NormalizedMax(children):
                new_children = [walk_expr(c) for c in children]
                if all(nc is c for nc, c in zip(new_children, children)):
                    return f
                return NormalizedMax(frozenset(new_children))
            case NormalizedGather(source, dim_in_source, idx):
                new_source = walk_expr(source)
                new_idx = walk_expr(idx)
                if new_source is source and new_idx is idx:
                    return f
                return NormalizedGather(new_source, dim_in_source, new_idx)
            case NormalizedScatter(source, dim_in_dest, idx):
                new_source = walk_expr(source)
                new_idx = walk_expr(idx)
                if new_source is source and new_idx is idx:
                    return f
                return NormalizedScatter(new_source, dim_in_dest, new_idx)
            case NormalizedParametricReduce(loop_var, lo, hi, op, body):
                new_body = walk_expr(body)
                if new_body is body:
                    return f
                return NormalizedParametricReduce(loop_var, lo, hi, op, new_body)
            case _:
                return f

    def walk_product(p):
        items = []
        changed = False
        for f, count in p.factors.items():
            new_f = walk_factor(f)
            items.append((new_f, count))
            if new_f is not f:
                changed = True
        if not changed:
            return p
        return NormalizedProduct(
            const=p.const,
            factors=FrozenCounter.from_dict(dict(items)),
        )

    def walk_expr(e):
        new_num = walk_product(e.num)
        new_den = walk_product(e.den)
        if new_num is e.num and new_den is e.den:
            return e
        return make_expr(new_num, new_den)

    return walk_expr(expr)


def _try_factor_common_gather(
    varying_factors : "FrozenCounter[NormalizedFactor]",
    reduce_dim : FullDim,
    op : ReduceOpType,
) -> "tuple[FullDim, NormalizedExpr, NormalizedExpr] | None":
    """
    If every factor is a `NormalizedGather` sharing the same
    `(dim_in_source, idx)` and `reduce_dim` isn't in the gather's
    output dims (`idx` doesn't vary with `reduce_dim`), return
    `(dim_in_source, idx, inner_body)` for the gather-through-reduce
    rewrite. `inner_body` is the product of the gathers' sources,
    which becomes the body of the post-rewrite reduce. Returns `None`
    otherwise.
    """
    if not varying_factors:
        return None
    common_dim_in_source : FullDim | None = None
    common_idx : "NormalizedExpr | None" = None
    for f in varying_factors:
        if not isinstance(f, NormalizedGather):
            return None
        if common_dim_in_source is None:
            common_dim_in_source = f.dim_in_source
            common_idx = f.idx
        elif (
            f.dim_in_source != common_dim_in_source or f.idx != common_idx
        ):
            return None
    assert common_idx is not None and common_dim_in_source is not None
    if varies_with_dim(common_idx, reduce_dim):
        # `reduce_dim` is part of the gather's output (in idx's shape);
        # can't factor the gather out — the reduce *is* over the
        # gathered dim.
        return None
    inner_body = NormalizedExpr.of(NormalizedProduct(const=1.0))
    for f, count in varying_factors.items():
        for _ in range(count):
            inner_body = mul(inner_body, f.source)
    return common_dim_in_source, common_idx, inner_body


def _as_gather_factor(expr : "NormalizedExpr") -> "NormalizedGather | None":
    """If `expr` is exactly one `NormalizedGather` factor (no const, no
    other factors, no denominator), return it. Used by Scatter's
    round-trip rewrite to recognize a wrapped Gather."""
    if (
        expr.num.const != 1.0
        or expr.den.factors
        or expr.den.const != 1.0
    ):
        return None
    factor = _as_single_factor(expr.num)
    return factor if isinstance(factor, NormalizedGather) else None


def _as_scatter_factor(expr : "NormalizedExpr") -> "NormalizedScatter | None":
    """Mirror of `_as_gather_factor` for `NormalizedScatter`."""
    if (
        expr.num.const != 1.0
        or expr.den.factors
        or expr.den.const != 1.0
    ):
        return None
    factor = _as_single_factor(expr.num)
    return factor if isinstance(factor, NormalizedScatter) else None


def _idx_has_property(norm_idx : "NormalizedExpr", prop : str) -> bool:
    """True iff `norm_idx` is exactly a bare `NormalizedTensor` whose
    name has been declared with `prop` via `runtime_index(...,
    permutation=True)` or similar. Sliced indices and other
    transformations don't carry the property forward — only the bare
    declared tensor counts (a sliced permutation isn't necessarily a
    permutation of the whole)."""
    if (
        norm_idx.num.const != 1.0
        or norm_idx.den.factors
        or norm_idx.den.const != 1.0
    ):
        return False
    factor = _as_single_factor(norm_idx.num)
    if not isinstance(factor, NormalizedTensor) or factor.tag is not None:
        return False
    return index_has_property(factor.name, prop)


@functools.cache
def normalize(expr : ExprType) -> NormalizedExpr:
    match expr:
        case Constant(x):
            const = NormalizedProduct(const=x)
            return NormalizedExpr.of(const)
        case Tensor(dims=dims, tag=tag, name=name):
            norm_tag = _normalize_tag(tag) if tag is not None else None
            t = NormalizedTensor(dims=frozenset(dims), tag=norm_tag, name=name)
            return NormalizedExpr.of(t)
        case UnaryOp(op, child):
            return unary_op(op, normalize(child))
        case BinaryOp(op, lhs, rhs):
            return binary_op(op, normalize(lhs), normalize(rhs))
        case Repeat(dim, child):
            # Repeat is special - don't pre-normalize! Slice info on
            # `dim` is structural noise (the broadcast doesn't care
            # about tile bounds); strip to FullDim so a tile-restricted
            # spec and a tile-walked kernel canonicalize identically.
            return repeat(frozenset([dim_full_dim(dim)]), child)
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
        case Gather(source, dim_in_source, idx):
            # Slice info on `dim_in_source` is structural noise — the
            # gather itself reads opaquely along the named axis, and any
            # tile-restriction lives in the enclosing shape, not in the
            # gather's identity. Strip down to FullDim for canonical form.
            dim_in_source = dim_full_dim(dim_in_source)
            norm_source = normalize(source)
            norm_idx = normalize(idx)
            # Permutation round-trip: `gather(scatter(Y, dim, perm), dim, perm) = Y`
            # when `perm` is a permutation. The scatter populated each
            # position of its output exactly once via `perm`; reading
            # back through `gather` with the same `perm` lands at the
            # same positions, recovering `Y`.
            inner = _as_scatter_factor(norm_source)
            if (
                inner is not None
                and inner.dim_in_dest == dim_in_source
                and inner.idx == norm_idx
                and _idx_has_property(norm_idx, "permutation")
            ):
                return inner.source
            # Gather-of-gather hoist: `gather(B, dim_B, gather(C, dim_C, π))`
            # = `gather(gather(B, dim_B, C), dim_C, π)`. The outer `π`
            # gets lifted out of the inner indexing — same values,
            # different factorization. This is the rewrite that makes
            # sort-based MoE verifiable: after argsort `π`,
            # `gather(W, gather(eid, π))` canonicalizes to
            # `gather(gather(W, eid), π)`, so the surrounding einsum
            # sees `π` as a single uniform outer indexing on every
            # n_tokens-shaped factor.
            inner_gather = _as_gather_factor(norm_idx)
            if (
                inner_gather is not None
                and inner_gather.dim_in_source != dim_in_source
            ):
                return NormalizedExpr.of(NormalizedGather(
                    source=NormalizedExpr.of(NormalizedGather(
                        source=norm_source,
                        dim_in_source=dim_in_source,
                        idx=inner_gather.source,
                    )),
                    dim_in_source=inner_gather.dim_in_source,
                    idx=inner_gather.idx,
                ))
            return NormalizedExpr.of(NormalizedGather(
                source=norm_source,
                dim_in_source=dim_in_source,
                idx=norm_idx,
            ))
        case Scatter(source, dim_in_dest, idx):
            dim_in_dest = dim_full_dim(dim_in_dest)
            norm_source = normalize(source)
            norm_idx = normalize(idx)
            # Dual round-trip: `scatter(gather(Y, dim, perm), dim, perm) = Y`
            # when `perm` is a permutation.
            inner = _as_gather_factor(norm_source)
            if (
                inner is not None
                and inner.dim_in_source == dim_in_dest
                and inner.idx == norm_idx
                and _idx_has_property(norm_idx, "permutation")
            ):
                return inner.source
            return NormalizedExpr.of(NormalizedScatter(
                source=norm_source,
                dim_in_dest=dim_in_dest,
                idx=norm_idx,
            ))


def verify_exprs_equivalent(x : ExprType, y : ExprType) -> bool:
    # Bound-subsumption uses the active `LoopScope`s and free dim
    # natural ranges (from `g_dim_registry`) — both runtime context the
    # cached `normalize` can't see. Apply post-normalize so a rolled
    # causal flash that stops at `k = qctx_size/BN` (skipping nctx
    # tiles past the diagonal) matches its full softmax-attention
    # spec: `n <= q` subsumes both `n < BN*K` and `n < nctx_size`
    # when `qctx_size ≤ min(BN*K, nctx_size)`, so both sides
    # canonicalize to the same `{n ≥ 0, n <= q}`-only form.
    return (
        simplify_under_active_loop_scope(normalize(x))
        == simplify_under_active_loop_scope(normalize(y))
    )

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
