import functools
import math
from dataclasses import dataclass, field

from .type import *
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
class NormalizedTensor:
    dims : frozenset[FullDim]

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash(self.dims)
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
    dim : FullDim
    op : ReduceOpType
    intervals : tuple[tuple[int, int], ...]
    child : "NormalizedExpr"

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash((self.dim, self.op, self.intervals, self.child))
        object.__setattr__(self, '_h', h)
        return h

def _canonicalize_intervals(intervals : tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    """Sort and merge adjacent/overlapping intervals."""
    if not intervals:
        return ()
    sorted_ints = sorted(intervals)
    merged = [sorted_ints[0]]
    for start, end in sorted_ints[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)

def make_reduce(
    dim : FullDim,
    op : ReduceOpType,
    intervals : tuple[tuple[int, int], ...],
    child : "NormalizedExpr",
) -> NormalizedReduce:
    """Canonical factory for NormalizedReduce. Always use this instead of the
    raw constructor so the intervals end up sorted and merged."""
    return NormalizedReduce(dim, op, _canonicalize_intervals(intervals), child)

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

def _merge_sum_reduces(terms : list["NormalizedProduct"]) -> list["NormalizedProduct"]:
    """Combine terms that are `const * sum-Reduce(...)` sharing const, dim, and child
    by unioning their intervals."""
    groups : dict[tuple[float, FullDim, "NormalizedExpr"], list[tuple[int, int]]] = {}
    others : list[NormalizedProduct] = []
    for t in terms:
        factor = _as_single_factor(
            NormalizedProduct(const=1.0, factors=t.factors),
        )
        if (
            isinstance(factor, NormalizedReduce)
            and factor.op == "sum"
        ):
            key = (t.const, factor.dim, factor.child)
            groups.setdefault(key, []).extend(factor.intervals)
        else:
            others.append(t)

    merged : list[NormalizedProduct] = []
    for (const, dim, child), intervals in groups.items():
        reduce_factor = make_reduce(dim, "sum", tuple(intervals), child)
        merged.append(NormalizedProduct(
            const=const,
            factors=FrozenCounter.from_iterable([reduce_factor]),
        ))

    return others + merged

def _is_pure_const(expr : "NormalizedExpr") -> bool:
    return not expr.num.factors and not expr.den.factors

def make_max(children) -> "NormalizedExpr":
    """Build a canonical NormalizedMax. Flattens nested maxes, drops -inf terms
    (identity for max), merges max-Reduce children that share dim and child by
    unioning their intervals, deduplicates, and collapses singletons.
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

    groups : dict[tuple[FullDim, NormalizedExpr], list[tuple[int, int]]] = {}
    others : list[NormalizedExpr] = []
    for c in flat:
        single = None
        if not c.den.factors and c.num.const == 1.0 and c.den.const == 1.0:
            single = _as_single_factor(c.num)
        if isinstance(single, NormalizedReduce) and single.op == "max":
            key = (single.dim, single.child)
            groups.setdefault(key, []).extend(single.intervals)
            continue
        others.append(c)

    for (d, child), intervals in groups.items():
        merged = make_reduce(d, "max", tuple(intervals), child)
        others.append(NormalizedExpr.of(merged))

    unique = frozenset(others)
    if not unique:
        return NormalizedExpr.of(NormalizedProduct(const=float("-inf")))
    if len(unique) == 1:
        return next(iter(unique))
    return NormalizedExpr.of(NormalizedMax(unique))

def make_sum(terms) -> "NormalizedProduct":
    """Build a canonical sum-of-products, returned as a NormalizedProduct.
    Flattens nested NormalizedSums, drops zero terms, combines like terms,
    merges disjoint sum-Reduce intervals, extracts common factors, and collapses
    singletons.
    """
    flat = _flatten_sum_terms(terms)
    flat = [t for t in flat if t.const != 0.0]
    flat = _merge_sum_reduces(flat)

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

NormalizedFactor = NormalizedTensor | NormalizedExp | NormalizedUnaryOp | NormalizedSum | NormalizedMax | NormalizedRepeat | NormalizedReduce

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
    hoists Repeat(D, ...)-invariant factors out of Repeats, enforces den.const == 1.0.
    """
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

def add(lhs : NormalizedExpr, rhs : NormalizedExpr):
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
    Collapse exp(0) = 1. Otherwise wrap unchanged as NormalizedExp.
    """
    if _is_pure_const(nchild) and nchild.num.const == 0.0:
        return NormalizedExpr.of(NormalizedProduct(const=1.0))

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
        exp_factor = NormalizedExp(term_expr)
        target = num_factors if term.const >= 0 else den_factors
        target[exp_factor] = target.get(exp_factor, 0) + 1

    num = NormalizedProduct(1.0, FrozenCounter.from_dict(num_factors))
    den = NormalizedProduct(1.0, FrozenCounter.from_dict(den_factors))
    return make_expr(num, den)


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
        case NormalizedReduce(d, op, intervals, child):
            return NormalizedExpr.of(
                make_reduce(d, op, intervals, strip_repeats_from_expr(child, dim))
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


def reduce(dim : FullDim, op : ReduceOpType, interval : tuple[int, int], child : NormalizedExpr):
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
    reduction = make_reduce(
        dim=dim_full_dim(dim),
        op=op,
        intervals=(interval,),
        child=expr_to_reduce,
    )
    reduction_expr = NormalizedExpr.of(reduction)
    return div(mul(invariant_num_expr, reduction_expr), invariant_den_expr)


@functools.cache
def normalize(expr : ExprType) -> NormalizedExpr:
    match expr:
        case Constant(x):
            const = NormalizedProduct(const=x)
            return NormalizedExpr.of(const)
        case Tensor(dims):
            t = NormalizedTensor(dims=frozenset(dims))
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
