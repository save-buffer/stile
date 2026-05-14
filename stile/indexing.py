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
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class SymbolicInt:
    """
    A symbolic integer atom — the unit out of which `AffineExpr`s are
    built. Identity is structural: two `SymbolicInt`s are equal iff
    their `name` and `source` both match.

    `source`, when set, encodes that the atom represents a runtime
    tensor-element lookup: the pair `(tensor_name, position)` means
    "the value of `tensor_name` at `position`" (e.g. `offsets[g]`).
    Two `tensor_element` lookups at the same position produce
    field-equal `SymbolicInt`s automatically — no string-synthesis
    needed. Bare atoms (reduce indices, free invariant vars, loop
    binders) have `source=None`.

    Non-identity metadata (declared min/max bounds, etc.) lives in the
    `_g_symint_metadata` side registry, looked up via
    `symint_info(atom)`. Bounds are advisory and don't affect
    equality; sources do.

    `runtime_value` is the execution-mode binding: a Python int or a
    jax tracer carrying this atom's concrete value during a
    `tjax.jit`-traced execution. Excluded from equality and hashing —
    two SymbolicInts with the same `name`/`source` are the same atom
    to the verifier whether or not one carries a tracer. tjax ops
    check this field via `_bound_runtime` to decide between the
    symbolic, concrete-int, and jax-tracer execution paths.
    """
    name : str
    source : "tuple[str, AffineExpr] | None" = None
    runtime_value : object = field(default=None, compare=False, hash=False, repr=False)

    def __add__(self, other) -> "AffineExpr": return to_affine(self) + other
    def __radd__(self, other) -> "AffineExpr": return to_affine(other) + to_affine(self)
    def __sub__(self, other) -> "AffineExpr": return to_affine(self) - other
    def __rsub__(self, other) -> "AffineExpr": return to_affine(other) - to_affine(self)
    def __mul__(self, k : int) -> "AffineExpr": return to_affine(self) * k
    def __rmul__(self, k : int) -> "AffineExpr": return to_affine(self) * k
    def __neg__(self) -> "AffineExpr": return -to_affine(self)


# Backward-compat alias — pre-rename, this was the only name for the
# atom and is still used by a lot of code and tests. Now just an alias
# for `SymbolicInt`. Can be removed once all call sites migrate.
LoopVariable = SymbolicInt


@dataclass(frozen=True)
class AffineExpr:
    """
    Canonical affine expression: `const + sum(coeff * var)`. The `terms` set
    never contains a pair with a zero coefficient, so equality and hashing
    give the algebraic equivalence relation.
    """
    const : int
    terms : frozenset[tuple[SymbolicInt, int]]

    def __post_init__(self):
        for _, c in self.terms:
            assert c != 0, "AffineExpr invariant: no zero coefficients"

    def __add__(self, other) -> "AffineExpr":
        other = to_affine(other)
        new_const = self.const + other.const
        merged : dict[SymbolicInt, int] = {v : c for v, c in self.terms}
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


@dataclass(frozen=True)
class SymInfo:
    """
    Declared *non-identity* metadata for a `SymbolicInt`, looked up via
    `symint_info(atom)`. Currently just bounds; future kinds
    (monotonicity, parity, …) slot in as new optional fields.

    Identity-bearing metadata (specifically `source` — "this atom IS
    the value of `tensor[position]`") lives on the `SymbolicInt`
    itself, not here, because it discriminates between distinct atoms.
    Bounds, by contrast, are advisory and don't change which atom
    you're talking about.
    """
    min : int | None = None
    max : int | None = None


# Bounds-and-such for SymbolicInts, keyed by the atom itself (full
# identity — name + source — so two distinct atoms with the same
# `name` but different `source`s don't share a registry slot).
_g_symint_metadata : dict["SymbolicInt", SymInfo] = {}


def symint_info(atom : "SymbolicInt") -> SymInfo | None:
    """
    Declared metadata for `atom`, or `None` if no factory has registered any.
    """
    return _g_symint_metadata.get(atom)


def _register_symint(atom : "SymbolicInt", info : SymInfo) -> None:
    """
    Add `info` to the registry for `atom`. If already declared, fields
    must be field-by-field compatible: each is either equal or one of
    (existing, new) is `None`.
    """
    existing = _g_symint_metadata.get(atom)
    if existing is None:
        _g_symint_metadata[atom] = info
        return
    merged_kwargs : dict = {}
    for field_name in ("min", "max"):
        a = getattr(existing, field_name)
        b = getattr(info, field_name)
        if a is None:
            merged_kwargs[field_name] = b
        elif b is None or a == b:
            merged_kwargs[field_name] = a
        else:
            raise ValueError(
                f"Conflicting {field_name} for SymbolicInt {atom!r}: "
                f"existing={a}, new={b}"
            )
    _g_symint_metadata[atom] = SymInfo(**merged_kwargs)


def runtime_scalar(name : str, max_value : int, min_value : int = 0) -> SymbolicInt:
    """
    A runtime-known integer in `[min_value, max_value)`. Returns a
    bare `SymbolicInt(name)` (no `source`) and registers its bounds
    in `_g_symint_metadata` so the verifier reads them during
    natural-range subsumption. Use as a `fori_loop` upper bound or
    anywhere a `SymbolicIndex` is expected.
    """
    atom = SymbolicInt(name)
    _register_symint(atom, SymInfo(min=min_value, max=max_value))
    return atom


def runtime_scalar_max(atom_or_name) -> int | None:
    """
    Upper bound declared for the atom, or `None` if undeclared.
    Verifier calls this during natural-range subsumption when a free
    `SymbolicInt` isn't in any active `LoopScope` nor
    `g_dim_registry`. Accepts either a `SymbolicInt` or a bare name
    string (for the bare-atom case where source is `None` — keeps
    old call sites that pass `v.name` working).
    """
    if isinstance(atom_or_name, str):
        atom = SymbolicInt(atom_or_name)
    else:
        atom = atom_or_name
    info = _g_symint_metadata.get(atom)
    return info.max if info is not None else None


def tensor_element(tensor_name : str, position) -> SymbolicInt:
    """
    A `SymbolicInt` representing the runtime value of `tensor_name`'s
    element at `position` (e.g. `offsets[g]`). The atom carries
    `source=(tensor_name, position)` as part of its identity, so two
    lookups at the same position yield field-equal atoms and two
    lookups at different positions yield distinct atoms — no string-
    synthesis needed. The verifier's block-constant rewrite reads
    `atom.source` to recognize when a slice is bounded by
    `(offsets[g], offsets[g+1])`.
    """
    pos_aff = to_affine(position)
    return SymbolicInt(name=tensor_name, source=(tensor_name, pos_aff))


# Backward-compat shim — pre-refactor, `RuntimeScalar` was a class.
# Now it's just `runtime_scalar(...)` returning a `SymbolicInt`.
def RuntimeScalar(name : str, max_value : int) -> SymbolicInt:
    return runtime_scalar(name, max_value)


# Legacy registry name — empty; kept so `reset_stile()`'s old
# `_g_runtime_scalars.clear()` line still works. Drop with the
# `RuntimeScalar` shim.
_g_runtime_scalars : dict[str, SymbolicInt] = {}


# --- Index properties -----------------------------------------------------
# A runtime index tensor (e.g. `page_table`, `expert_id`) can carry
# declared algebraic properties — currently `"permutation"` (bijection
# from input dim to values_in dim) and `"partition"` (each input mapped
# to exactly one value in values_in). The verifier reads these during
# property-driven rewrites: e.g. `scatter(gather(Y, perm), perm) = Y`
# when `perm` is a permutation. Properties are name-keyed because the
# user holds the typed-tensor handle but the verifier sees the bare
# `NormalizedTensor` by name. `reset_stile()` clears the registry.
_g_index_properties : dict[str, frozenset[str]] = {}

def declare_index_properties(name : str, *properties : str) -> None:
    """Register `properties` on the runtime-index tensor named `name`.
    Recognized: `"permutation"`, `"partition"`. Idempotent for the same
    set; conflicting later calls raise."""
    new = frozenset(properties)
    existing = _g_index_properties.get(name)
    if existing is not None and existing != new:
        raise ValueError(
            f"Conflicting properties for index {name!r}: existing "
            f"{set(existing)}, new {set(new)}"
        )
    _g_index_properties[name] = new


def index_has_property(name : str, prop : str) -> bool:
    """True iff the named runtime index has been declared with `prop`."""
    return prop in _g_index_properties.get(name, frozenset())


# Pairing between a block-sorted index and its offsets tensor. When
# `eid_sorted` is declared paired with `offsets`, the verifier knows:
# for every block `g`, positions `[offsets[g], offsets[g+1])` of
# `eid_sorted` all equal `g`. This is what powers the block-constant
# rewrite: a `gather(W, eid_sorted)` inside a slice bounded by
# `(offsets[g], offsets[g+1])` collapses to `W[g]`.
_g_block_pairings : dict[str, str] = {}  # offsets_name → paired_idx_name


def declare_block_pairing(offsets_name : str, paired_idx_name : str) -> None:
    """
    Register that `paired_idx_name` is block-sorted with respect to
    `offsets_name`: positions `[offsets_name[g], offsets_name[g+1])` of
    the paired index all hold value `g`.
    """
    existing = _g_block_pairings.get(offsets_name)
    if existing is not None and existing != paired_idx_name:
        raise ValueError(
            f"Conflicting block pairing for offsets {offsets_name!r}: "
            f"existing paired with {existing!r}, new {paired_idx_name!r}"
        )
    _g_block_pairings[offsets_name] = paired_idx_name


def paired_index_for_offsets(offsets_name : str) -> str | None:
    """
    Name of the block-sorted index paired with `offsets_name`, or `None`
    if no pairing is declared. Used by the block-constant rewrite to
    recognize that a slice bounded by `(offsets[g], offsets[g+1])` lies
    within a single block of the paired index.
    """
    return _g_block_pairings.get(offsets_name)


# Boundary declarations for tensors used as offsets. A boundary
# declaration says: `tensor_name[position] == concrete_value` for one
# specific position. Used by `TypedResult.done()` to resolve symbolic
# slice bounds at the loop's first/last iteration into concrete values
# for coverage checking. E.g. declaring `offsets[0] == 0` and
# `offsets[n_experts] == n_tokens` lets `done()` verify that a
# fori_loop over experts produces blocks that tile `[0, n_tokens)`.
_g_tensor_boundaries : dict[tuple[str, int], int] = {}


def declare_tensor_boundary(
    tensor_name : str, position : int, value : int,
) -> None:
    """
    Declare that `tensor_name[position]` equals `value` (a compile-time
    int). The verifier uses this to substitute `tensor_element(tensor,
    position)` with `value` during coverage checks. Idempotent for the
    same `(position, value)`; conflicting declarations raise.
    """
    key = (tensor_name, position)
    existing = _g_tensor_boundaries.get(key)
    if existing is not None and existing != value:
        raise ValueError(
            f"Conflicting boundary declaration for {tensor_name}[{position}]: "
            f"existing={existing}, new={value}"
        )
    _g_tensor_boundaries[key] = value


def tensor_boundary(tensor_name : str, position : int) -> int | None:
    """
    Declared boundary value for `tensor_name[position]`, or `None` if
    undeclared.
    """
    return _g_tensor_boundaries.get((tensor_name, position))


def resolve_symbolic_index(idx) -> "SymbolicIndex":
    """
    Walk `idx` and replace any `tensor_element` atom whose
    `(tensor_name, position)` is a declared boundary with the declared
    integer value. Returns the resolved `SymbolicIndex`; concrete ints
    are returned as plain ints. Used by `TypedResult.done()` to
    materialize boundary slice bounds for coverage checking — the
    user-friendly version of "evaluate `offsets[0]` to `0` knowing the
    declaration".
    """
    aff = to_affine(idx)
    result_const = aff.const
    new_terms : dict[SymbolicInt, int] = {}
    for v, c in aff.terms:
        if v.source is not None:
            tensor_name, position = v.source
            pos_aff = to_affine(position)
            if not pos_aff.terms:
                pos_int = pos_aff.const
                boundary = tensor_boundary(tensor_name, pos_int)
                if boundary is not None:
                    result_const += c * boundary
                    continue
        # No boundary substitution — keep this term.
        new_terms[v] = new_terms.get(v, 0) + c
    return AffineExpr(
        result_const,
        frozenset((v, c) for v, c in new_terms.items() if c != 0),
    )


SymbolicIndex = AffineExpr | SymbolicInt | int


def to_affine(x : SymbolicIndex) -> AffineExpr:
    """
    Promote any `SymbolicIndex` to its canonical `AffineExpr` form.
    """
    if isinstance(x, AffineExpr):
        return x
    if isinstance(x, SymbolicInt):
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
    intervals = list(intervals)
    disjuncts = frozenset(
        _conjunction({ge(var, s), lt(var, e)}) for s, e in intervals
    )
    # Include any free LoopVariables that flow in via symbolic interval
    # bounds — `Domain.variables` must mention every variable referenced
    # by any constraint, otherwise two domains with the same constraints
    # but different `variables` sets compare unequal.
    free_vars = frozenset(
        v
        for (s, e) in intervals
        for x in (s, e)
        for (v, _) in to_affine(x).terms
    )
    return Domain(frozenset({var}) | free_vars, disjuncts)


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


# --- Rolled loops ---------------------------------------------------------

_active_loop_scopes : list["LoopScope"] = []


class LoopScope:
    """
    Context manager for a rolled loop. Enters the `with` block once with a
    symbolic `LoopVariable` bound to its iteration range; the body traces a
    single parametric instance, and the verifier proves the result holds for
    every integer value of the loop variable in `[start, end)`.
    """

    def __init__(self, name : str, start : SymbolicIndex, end : SymbolicIndex):
        self.var = LoopVariable(name)
        self.lo = start
        self.hi = end
        self.domain = range_domain(self.var, start, end)

    def __enter__(self) -> LoopVariable:
        _active_loop_scopes.append(self)
        return self.var

    def __exit__(self, *_exc):
        popped = _active_loop_scopes.pop()
        assert popped is self


def loop(name : str, start : SymbolicIndex, end : SymbolicIndex) -> LoopScope:
    """
    Create a rolled-loop scope. Use as `with stile.loop("i", 0, N) as i: ...`.
    Inside the block, `i` is a symbolic `LoopVariable` that can participate
    in affine arithmetic (e.g., `i * tile_size`, `i + 1`) to parameterize
    slice bounds and iteration structure.
    """
    return LoopScope(name, start, end)


def active_loop_domain() -> Domain | None:
    """
    The intersection of all currently-active `LoopScope` domains, or `None`
    if no loop is active. Used by the verifier to know what iteration domain
    a traced expression is parametric over.
    """
    if not _active_loop_scopes:
        return None
    variables : set[LoopVariable] = set()
    constraints : set = set()
    for scope in _active_loop_scopes:
        variables |= scope.domain.variables
        constraints |= scope.domain.constraints
    return Domain(frozenset(variables), frozenset(constraints))


def active_loop_vars() -> frozenset[LoopVariable]:
    """
    The set of `LoopVariable`s bound by all currently-active `LoopScope`s.
    Used by the verifier to distinguish symbolic bounds that represent
    loop iteration restrictions (which should sit in a reduce's interval
    bound) from cross-variable predicates (which should sit in extras).
    """
    return frozenset(s.var for s in _active_loop_scopes)
