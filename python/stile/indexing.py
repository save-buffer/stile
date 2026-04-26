"""
Symbolic indexing and polyhedral iteration domains.

A `SymbolicIndex` is an integer-valued affine expression over `LoopVariable`s.
It's stored in canonical form as an `AffineExpr` (`const + sum(coeff * var)`),
so two affine expressions compare equal iff they represent the same function
— no matter how they were built. `LoopVariable`s and plain `int`s are accepted
anywhere a `SymbolicIndex` is expected and get promoted to `AffineExpr` via
`to_affine`.

A `Domain` is a set of `LoopVariable`s plus a conjunction of `AffineConstraint`s
(each of the form `expr >= 0`) over them — the polyhedral iteration set.
`le`/`lt`/`ge`/`gt`/`eq` encode common inequality shapes into the canonical
form.

Operator overloads on `LoopVariable` and `AffineExpr` make ordinary Python
arithmetic produce `AffineExpr`s: `2 * i + 5`, `i - j`, `-(i)` all work.
Scalar multiplication requires a compile-time `int`.
"""
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LoopVariable:
    """
    Identity of a symbolic integer loop variable. Two LoopVariables are equal
    iff their names match, so within a single scope give each variable a
    distinct name. Bounds and other per-variable constraints live in a
    `Domain`, not here.
    """
    name : str

    def __add__(self, other) -> "AffineExpr": return to_affine(self) + other
    def __radd__(self, other) -> "AffineExpr": return to_affine(other) + to_affine(self)
    def __sub__(self, other) -> "AffineExpr": return to_affine(self) - other
    def __rsub__(self, other) -> "AffineExpr": return to_affine(other) - to_affine(self)
    def __mul__(self, k : int) -> "AffineExpr": return to_affine(self) * k
    def __rmul__(self, k : int) -> "AffineExpr": return to_affine(self) * k
    def __neg__(self) -> "AffineExpr": return -to_affine(self)


@dataclass(frozen=True)
class AffineExpr:
    """
    Canonical affine expression: `const + sum(coeff * var)`. The `terms` set
    never contains a pair with a zero coefficient, so equality and hashing
    give the algebraic equivalence relation.
    """
    const : int
    terms : frozenset[tuple[LoopVariable, int]]

    def __post_init__(self):
        for _, c in self.terms:
            assert c != 0, "AffineExpr invariant: no zero coefficients"

    def __add__(self, other) -> "AffineExpr":
        other = to_affine(other)
        new_const = self.const + other.const
        merged : dict[LoopVariable, int] = {v : c for v, c in self.terms}
        for v, c in other.terms:
            merged[v] = merged.get(v, 0) + c
        new_terms = frozenset((v, c) for v, c in merged.items() if c != 0)
        return AffineExpr(new_const, new_terms)

    __radd__ = __add__

    def __neg__(self) -> "AffineExpr":
        return AffineExpr(-self.const, frozenset((v, -c) for v, c in self.terms))

    def __sub__(self, other) -> "AffineExpr":
        return self + (-to_affine(other))

    def __rsub__(self, other) -> "AffineExpr":
        return to_affine(other) - self

    def __mul__(self, k : int) -> "AffineExpr":
        if not isinstance(k, int):
            raise TypeError(
                "AffineExpr can only be multiplied by a compile-time int; "
                f"got {type(k).__name__}"
            )
        if k == 0:
            return AffineExpr(0, frozenset())
        return AffineExpr(
            k * self.const,
            frozenset((v, k * c) for v, c in self.terms),
        )

    __rmul__ = __mul__


SymbolicIndex = AffineExpr | LoopVariable | int


def to_affine(x : SymbolicIndex) -> AffineExpr:
    """
    Promote any `SymbolicIndex` to its canonical `AffineExpr` form.
    """
    if isinstance(x, AffineExpr):
        return x
    if isinstance(x, LoopVariable):
        return AffineExpr(0, frozenset({(x, 1)}))
    if isinstance(x, int):
        return AffineExpr(x, frozenset())
    raise TypeError(f"Not a SymbolicIndex: {type(x).__name__}")


# ---- Domains and constraints ----------------------------------------------

@dataclass(frozen=True)
class AffineConstraint:
    """
    An affine inequality `expr >= 0` over the integers. Strict and equality
    constraints are encoded in this form by the `lt`/`gt`/`eq` helpers, so
    the canonical store has only one kind.
    """
    expr : AffineExpr


Conjunction = frozenset[AffineConstraint]


@dataclass(frozen=True)
class Domain:
    """
    A polyhedral iteration domain in DNF: a union of conjunctions of affine
    inequalities over a shared set of `LoopVariable`s. The iteration set is
    the integer assignments to `variables` that satisfy at least one
    conjunction ("disjunct").

    A single-conjunct Domain represents the familiar "`all of these
    constraints hold`" shape — that's how `range_domain` produces a loop
    range, and how `and_constraints` composes constraints. Disjunction
    arises from `or_domains` and from mask predicates that combine via `or`
    (e.g., local-attention-band OR global-sink-positions).
    """
    variables : frozenset[LoopVariable]
    disjuncts : frozenset[Conjunction]


def _conjunction(constraints : Iterable[AffineConstraint]) -> Conjunction:
    return frozenset(constraints)


def le(a : SymbolicIndex, b : SymbolicIndex) -> AffineConstraint:
    """
    `a <= b` on the integers, encoded as `b - a >= 0`.
    """
    return AffineConstraint(to_affine(b) - to_affine(a))


def lt(a : SymbolicIndex, b : SymbolicIndex) -> AffineConstraint:
    """
    `a < b` on the integers, encoded as `b - a - 1 >= 0`.
    """
    return AffineConstraint(to_affine(b) - to_affine(a) - 1)


def ge(a : SymbolicIndex, b : SymbolicIndex) -> AffineConstraint:
    """
    `a >= b` on the integers, encoded as `a - b >= 0`.
    """
    return AffineConstraint(to_affine(a) - to_affine(b))


def gt(a : SymbolicIndex, b : SymbolicIndex) -> AffineConstraint:
    """
    `a > b` on the integers, encoded as `a - b - 1 >= 0`.
    """
    return AffineConstraint(to_affine(a) - to_affine(b) - 1)


def eq(a : SymbolicIndex, b : SymbolicIndex) -> tuple[AffineConstraint, AffineConstraint]:
    """
    `a == b`, returned as the pair `(a <= b, a >= b)`. Callers should splat
    into their constraint set.
    """
    return (le(a, b), ge(a, b))


def domain(
    variables : Iterable[LoopVariable],
    constraints : Iterable[AffineConstraint],
) -> Domain:
    """
    Shorthand for a single-conjunct Domain.
    """
    return Domain(
        frozenset(variables),
        frozenset({_conjunction(constraints)}),
    )


def range_domain(
    var : LoopVariable,
    start : SymbolicIndex,
    end : SymbolicIndex,
) -> Domain:
    """
    The 1D domain `{var : start <= var < end}`, as a single-conjunct Domain.
    """
    return Domain(
        frozenset({var}),
        frozenset({_conjunction({ge(var, start), lt(var, end)})}),
    )


def and_constraints(
    d : Domain,
    *extra : AffineConstraint,
) -> Domain:
    """
    Conjoin `extra` onto every disjunct of `d`. Extra constraints can
    reference variables already in `d.variables` or introduce new ones.
    """
    if not extra:
        return d
    new_disjuncts = frozenset(
        _conjunction(list(conj) + list(extra)) for conj in d.disjuncts
    )
    new_vars = d.variables | frozenset(
        v for c in extra for v, _ in c.expr.terms
    )
    return Domain(new_vars, new_disjuncts)


def or_domains(*domains : Domain) -> Domain:
    """
    Union of domains: the DNF whose disjuncts are the union of the inputs'.
    All inputs' variables are combined.
    """
    if not domains:
        return Domain(frozenset(), frozenset())
    variables : frozenset = frozenset()
    disjuncts : frozenset = frozenset()
    for d in domains:
        variables = variables | d.variables
        disjuncts = disjuncts | d.disjuncts
    return Domain(variables, disjuncts)


def and_domains(d1 : Domain, d2 : Domain) -> Domain:
    """
    Intersection of two DNF domains:
    `(A1 ∨ A2) ∩ (B1 ∨ B2) = (A1 ∧ B1) ∨ (A1 ∧ B2) ∨ (A2 ∧ B1) ∨ (A2 ∧ B2)`.
    Variable sets are unioned (each side may reference vars the other doesn't).
    """
    new_disjuncts = frozenset(
        _conjunction(list(c1) + list(c2))
        for c1 in d1.disjuncts
        for c2 in d2.disjuncts
    )
    return Domain(d1.variables | d2.variables, new_disjuncts)


def _simplify_conjunction_over_var(
    conj : Conjunction,
    var : LoopVariable,
) -> Conjunction:
    """
    Collapse redundant 1-D bounds on `var` in a conjunction. For every
    constraint of the form `var + c >= 0` (lower bound `var >= -c`) or
    `-var + c >= 0` (upper bound `var <= c`) whose non-`var` part is a
    plain constant, take the tightest pair. Constraints that reference
    `var` with a higher-magnitude coefficient or that mix `var` with
    other variables are passed through unchanged.
    """
    lower_const_bounds : list[int] = []
    upper_exclusive_const_bounds : list[int] = []
    other : list[AffineConstraint] = []
    for c in conj:
        aff = c.expr
        var_coeff = 0
        mixes_other_vars = False
        for v, co in aff.terms:
            if v == var:
                var_coeff = co
            else:
                mixes_other_vars = True
        if mixes_other_vars or var_coeff not in (1, -1):
            other.append(c)
            continue
        if var_coeff == 1:
            # `var + aff.const >= 0` → `var >= -aff.const`
            lower_const_bounds.append(-aff.const)
        else:
            # `-var + aff.const >= 0` → `var <= aff.const` → `var < aff.const + 1`
            upper_exclusive_const_bounds.append(aff.const + 1)

    result = list(other)
    if lower_const_bounds:
        result.append(ge(var, max(lower_const_bounds)))
    if upper_exclusive_const_bounds:
        result.append(lt(var, min(upper_exclusive_const_bounds)))
    return _conjunction(result)


def simplify_domain(d : Domain) -> Domain:
    """
    Remove redundant 1-D interval constraints per variable in each disjunct.
    For each disjunct, collapse multiple lower bounds on a single variable
    to the tightest, and likewise for upper bounds. Mixed-variable and
    higher-coefficient constraints are untouched — full polyhedral
    simplification is out of scope.
    """
    new_disjuncts : set[Conjunction] = set()
    for conj in d.disjuncts:
        simplified = conj
        for var in d.variables:
            simplified = _simplify_conjunction_over_var(simplified, var)
        new_disjuncts.add(simplified)
    return Domain(d.variables, frozenset(new_disjuncts))


def interval_domain(
    var : LoopVariable,
    intervals : Iterable[tuple[SymbolicIndex, SymbolicIndex]],
) -> Domain:
    """
    Build a 1D `Domain` from a list of half-open intervals: one conjunct
    per interval, each expressing `start <= var < end`. The union represents
    the set of positions covered by *any* of the intervals.
    """
    disjuncts = frozenset(
        _conjunction({ge(var, s), lt(var, e)}) for s, e in intervals
    )
    return Domain(frozenset({var}), disjuncts)


# ---- Analysis -------------------------------------------------------------

def free_vars(expr : SymbolicIndex) -> frozenset[LoopVariable]:
    """
    Every `LoopVariable` appearing with a nonzero coefficient in `expr`.
    """
    return frozenset(v for v, _ in to_affine(expr).terms)


def substitute(
    expr : SymbolicIndex,
    bindings : dict[LoopVariable, SymbolicIndex],
) -> AffineExpr:
    """
    Replace each `LoopVariable` key in `bindings` with its value throughout
    `expr`, returning the resulting canonical `AffineExpr`.
    """
    expr = to_affine(expr)
    result = AffineExpr(expr.const, frozenset())
    for v, c in expr.terms:
        replacement = to_affine(bindings[v]) if v in bindings else to_affine(v)
        result = result + replacement * c
    return result


def evaluate(expr : SymbolicIndex, bindings : dict[LoopVariable, int]) -> int:
    """
    Evaluate `expr` to a concrete integer; every free variable in `expr` must
    appear in `bindings`.
    """
    expr = to_affine(expr)
    total = expr.const
    for v, c in expr.terms:
        if v not in bindings:
            raise ValueError(f"LoopVariable {v.name!r} is not bound")
        total += c * bindings[v]
    return total


def constraint_holds(
    constraint : AffineConstraint,
    bindings : dict[LoopVariable, int],
) -> bool:
    """
    True iff `constraint.expr >= 0` under the given bindings.
    """
    return evaluate(constraint.expr, bindings) >= 0


def domain_contains(domain : Domain, bindings : dict[LoopVariable, int]) -> bool:
    """
    True iff `bindings` assigns an integer to every variable in `domain`
    and at least one disjunct is fully satisfied.
    """
    if not all(v in bindings for v in domain.variables):
        return False
    return any(
        all(constraint_holds(c, bindings) for c in conjunction)
        for conjunction in domain.disjuncts
    )
