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


@dataclass(frozen=True)
class Domain:
    """
    A polyhedral iteration domain: a set of `LoopVariable`s and a conjunction
    of `AffineConstraint`s over them. The iteration set is the integer
    assignments to `variables` that satisfy every constraint.

    A `LoopVariable` with bounds is just a 1-variable `Domain` (see
    `range_domain`), and fancier patterns — band-diagonal masks for local
    attention, block-triangular sparsity, alignment constraints — drop in as
    additional inequalities on the same variables.
    """
    variables : frozenset[LoopVariable]
    constraints : frozenset[AffineConstraint]


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
    Shorthand for `Domain(frozenset(variables), frozenset(constraints))`.
    """
    return Domain(frozenset(variables), frozenset(constraints))


def range_domain(
    var : LoopVariable,
    start : SymbolicIndex,
    end : SymbolicIndex,
) -> Domain:
    """
    The 1D domain `{var : start <= var < end}`.
    """
    return Domain(
        frozenset({var}),
        frozenset({ge(var, start), lt(var, end)}),
    )


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
    True iff `bindings` assigns an integer to every variable in `domain` and
    every constraint is satisfied.
    """
    if not all(v in bindings for v in domain.variables):
        return False
    return all(constraint_holds(c, bindings) for c in domain.constraints)
