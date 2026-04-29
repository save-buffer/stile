# Expression verification

Stile verifies that two user-written expressions are algebraically equal by
normalizing each to a canonical form and then comparing the two forms structurally:

```python
def verify_exprs_equivalent(x, y):
    return normalize(x) == normalize(y)
```

If every algebraic identity Stile is willing to believe in is applied during
`normalize`, and the canonical form is unique per equivalence class, then `==`
is a sound and complete decision procedure for the identities we know about.
This document describes what those identities are, where they live in the code,
and where the canonicalization is known to be incomplete.

## The canonical form

The output of `normalize` is a `NormalizedExpr = num / den` where `num` and
`den` are `NormalizedProduct`s. A `NormalizedProduct` is a scalar `const`
times a multiset of `NormalizedFactor`s. The multiset is stored as a
`FrozenCounter` (see `frozen_counter.py`) — an immutable, hashable multiset
with multiset-union (`+`), difference (`-`), min (`&`), and max (`|`).

```
NormalizedExpr       = NormalizedProduct / NormalizedProduct
NormalizedProduct    = float * multiset[NormalizedFactor]
NormalizedFactor     = NormalizedTensor
                     | NormalizedExp
                     | NormalizedUnaryOp
                     | NormalizedSum
                     | NormalizedMax
                     | NormalizedRepeat
                     | NormalizedReduce
```

The invariants of the canonical form are:

- `den.const == 1.0` (any numeric constant lives in `num.const`).
- Numerator and denominator share no factors (cancellation has been applied).
- Inside a `NormalizedRepeat(dims, child)`, every factor of `child` varies
  with at least one `d ∈ dims` — invariants have been hoisted out.
- `NormalizedReduce.intervals` is a sorted, merged tuple of disjoint ranges.
- `NormalizedSum` has no nested sums, no zero terms, no like terms
  (distinct term multisets), no common factor across all children, and no
  pair of sum-Reduces that share dim/child (they get their intervals unioned).
- `NormalizedMax` has no nested maxes, no `-inf` children, no pair of
  max-Reduces sharing dim/child, and no duplicate children.
- Unary ops on pure-constant arguments are evaluated.

## Where the rules live

Canonicalization happens at construction time. The raw frozen-dataclass
constructors are callable, but **every code path goes through a factory**
that enforces the invariants:

| Factory | Handles |
|---|---|
| `make_expr(num, den)` | den.const = 1 invariant; num/den cancellation; hoist invariants out of Repeat factors (on either side) |
| `make_sum(terms)` | flatten nested `NormalizedSum`s; drop zero-const terms; combine like terms; merge sum-Reduces over disjoint intervals; extract common factors (`A*B + A*C = A*(B+C)`); collapse singletons |
| `make_max(children)` | flatten nested `NormalizedMax`es; drop `-inf`; merge max-Reduces with shared dim/child; dedupe; collapse singletons |
| `make_reduce(dim, op, intervals, child)` | sort and merge adjacent intervals |
| `NormalizedExpr.of(x)` | lift product/factor/expr into a NormalizedExpr, routing through `make_expr` |

Operation-specific normalization lives in its own function:

- `add` / `sub` build `(ad ± bc) / bd` and route the numerator through `make_sum`.
- `mul` / `div` multiply/divide the products and route through `make_expr`.
- `unary_op` dispatches `exp` to `normalize_exp`; for `sqrt`, `sin`, `cos`,
  `exp` it evaluates pure-const children (`sqrt(16) → 4`).
- `normalize_exp` distributes exp across sums
  (`exp(a+b-c) = exp(a)*exp(b)/exp(c)`) and collapses `exp(0) → 1`.
- `repeat` is called on the un-normalized `ExprType` (not a normalized expr)
  so it can push `Repeat` through `UnaryOp`/`BinaryOp` and merge nested
  `Repeat`s before the child is normalized. It skips wrapping pure scalar
  constants, since `Repeat(D, c) = c` under implicit broadcasting.
- `reduce` partitions the child's factors by whether they vary with the
  reduction dim, runs `strip_repeats_from_product` on the invariant portion
  to remove redundant `Repeat(dim, …)` wrappers once `dim` has been
  eliminated, and stitches everything back together via `mul` / `div`.

## Dim-variance and the hoist/strip machinery

The fulcrum of the reduce/repeat interaction is
`varies_with_dim(factor, dim) : bool`, which answers "does the *value* of
this factor change as `dim` varies?" (not "does it mention `dim`"). By this
definition, `Repeat(dim, x)` is dim-**invariant** (constant along dim),
and `Reduce(dim, x)` is also dim-invariant (the dim has been eliminated).

Two dual rules depend on it:

- **Hoist out of reduce**: `sum_D(F(D) * G) = G * sum_D(F(D))` when `G` is
  D-invariant. Applied in `reduce()`.
- **Hoist out of repeat**: `Repeat(D, A_invar * G_vary) = A_invar * Repeat(D, G_vary)`.
  Applied in `make_expr` via `_hoist_invariants_from_repeat`.

We chose to hoist *out* rather than push *in* because pushing invariant
siblings into a `Repeat` during intermediate `mul`s blocks later
cancellation — different operation orders would produce different
canonical forms.

## Known gaps

These are cases where the normalizer is **incomplete**: two algebraically
equivalent expressions can normalize to different forms. None of these are
known to cause the normalizer to be **unsound** (accept non-equivalent
expressions); see `tests/test_normalization_inequivalence.py` for the
rejection tests.

- **Max with additive constants.** `max_N(c + f) = c + max_N(f)` is always
  sound but not implemented. Hoisting scaling constants, `max_N(c * f) = c * max_N(f)`,
  is *only* sound for `c ≥ 0` and isn't attempted either. The current
  `reduce()` hoists arbitrary dim-invariant factors out of both sum- and
  max-reduces; for max that's only sound when we happen to know the hoisted
  factor is non-negative (e.g., an `exp(...)` factor). This is fine for the
  softmax/flash-attention tests (all hoisted factors are `exp`s) but is a
  latent soundness risk for other inputs.
- **Sum-across-denominator cancellation.** `make_sum` extracts factors
  common to every term in a sum, which handles `A*B + A*C → A*(B+C)`. But
  something like `(A + B) / C + (A - B) / C` isn't merged into
  `2*A / C` — the two denominators aren't combined before the sum.
- **Nested fractions inside sum terms.** A `NormalizedSum`'s children are
  `NormalizedProduct`s, which have no denominator. Anything that would
  produce a per-term denominator is handled by `add`/`sub` giving the whole
  sum a common denominator, but this means sum simplifications that require
  different per-term denominators aren't expressible.
- **Sliced-dim tensors in Repeat.** `NormalizedRepeat.dims: frozenset[FullDim]`.
  If a user writes `Repeat(N[0:4], x)`, the slice is currently dropped at the
  factor level. `NormalizedReduce.intervals` handles the analogous case for
  reductions; Repeat would need the same treatment if sliced-Repeats become
  a real use case.
- **Integer `max` reductions.** `NormalizedMax` drops `-inf` as the identity,
  which assumes a float-valued context. Integer or other max semantics
  would need different identities.
- **No simplification of `x^0 = 1` or like-power collapsing.** We don't
  currently track factor exponents beyond multiplicity in a FrozenCounter,
  which is fine because all our factors appear with integer multiplicities.
  But `1/x * x^2 = x` only works because the FrozenCounter multiset math
  happens to implement it — we haven't thought systematically about
  non-integer powers.
- **No trig identities beyond constant evaluation.** `sin(x)^2 + cos(x)^2`
  won't collapse to `1`; such identities would need explicit rewrites.

When adding a new canonicalization rule, the litmus test is the rejection
suite in `test_normalization_inequivalence.py` — any rule you add has to
keep those tests rejecting.
