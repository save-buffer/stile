# Stile: A Type System for Numerical Programs

**Formally Verify Your Numerical Programs**

Stile is a type system for numerical programs. Describe what your program should compute in a lightweight specification language and Stile will structurally prove the program's adherance to the spec.

A short demo of what this lets you write today:

```python
# Spec: causal flash attention, one line.
SPEC = (
    "(softmax[nctx where nctx <= qctx]"
    "((qctx dhead, nctx dhead -> qctx nctx) / sqrt(16)), "
    "nctx dhead -> qctx dhead)"
)

# Kernel: tile-walk only the lower triangle, online softmax, bias mask
# on the diagonal tile. Per-qctx-tile structural proof.
for iqctx in range(0, qctx.size, qctx_tile_size):
    running_max, running_l, o = -jnp.inf, 0, 0
    for ictx in range(0, iqctx + qctx_tile_size, nctx_tile_size):
        q_tile = Q.slice(qctx, iqctx, iqctx + qctx_tile_size)
        k_tile = K.slice(nctx, ictx, ictx + nctx_tile_size)
        qk = tjax.einsum(q_tile, k_tile, "qctx dhead, nctx dhead -> nctx qctx") / jnp.sqrt(dhead.size)
        # Bias-form mask: 0 inside causal region, -inf outside.
        qk = qk + tjax.mask(qk.type.st, "nctx <= qctx", 0.0, -jnp.inf)
        # ... online-softmax accumulator over qk, V ...
    o.assert_equivalent(SPEC, nctx[:(iqctx + qctx_tile_size)])  # proof
```

The verifier proves the kernel's online softmax, with `-inf` bias on the diagonal tile and skipped upper-triangle tiles, normalizes to the explicit causal-attention spec.

## How it works

A Stile-typed tensor has two types:
- **`ShapeType`**: which logical dims the tensor has, and how each is sliced.
- **`ExprType`**: the AST of operations performed up to this point.

`ShapeType` is enforced eagerly: you can only multiply two tensors with matching dim signatures, you must reduce the entire dim before assigning the result, etc.

`ExprType` is checked at assignment time. The verifier normalizes both your kernel's accumulated `ExprType` and the spec's parsed `ExprType` into a canonical form and compares. The canonical form folds:
- algebraic identities (`exp(0) = 1`, `0 + x = x`, `0 * x = 0`, `exp(a-b) = exp(a)/exp(b)`, `exp(-inf) = 0`, …),
- distribution through tagged tensors (`*` and `+` push through `Cond(D, …)` branches),
- iteration-domain folding (`sum(body * Cond(D, 1, 0))` collapses to a sum over `[0, N) ∩ D`),
- adjacent-tile interval merging on sum and max reductions, including with shared cross-variable predicates,
- max push-through and `-inf` absorption (so bias-form masks converge with multiplicative-form masks),
- post-fold invariant hoisting (the piece that lets the kernel's `exp(max)` rescaling factor cancel between numerator and denominator).

If the two normalize to the same expression, they compute the same function.

## The specification language

A specification is a small expression over named dims. Tensors are written as the sequence of their dims (`Q D` is a tensor of shape `Q × D`). Slices use `D[a:b]`. Einsums use `,` to separate operands and `->` to give the output shape.

```
Q D K                                       # a tensor with three dims
Q D[0:8]                                    # tensor with D sliced to [0, 8)
2 * Q D                                     # scaled tensor
A B + A B                                   # addition (same-shape required)
exp(Q D)                                    # elementwise unary
(Q D, K D -> Q K)                           # einsum: sum over D
sum[K](Q K)                                 # explicit reduction
sum[K where K <= Q](Q K)                    # iteration-restricted sum (mult-mask)
max[K where K <= Q](Q K)                    # iteration-restricted max (bias-mask)
softmax[K](Q K)                             # softmax along K
softmax[K where K <= Q](Q K)                # causal softmax — restricts both num and den
sum[N](Q N where N >= 4) -> Q               # mult-mask `where`-clause inside sum
```

**`[d where P]`** restricts the iteration of a reduction by the affine predicate `P`. Lowers based on the surrounding op:
- `sum[d where P]` uses a multiplicative mask (`P` zero-elsewhere).
- `max[d where P]` and `softmax[d where P]` use a bias mask (`-inf` elsewhere) — masked positions vanish through `max`'s identity and `exp`'s zero, restricting both the numerator's exp and the denominator's sum.

**`body where P`** (outside a dim annotation) is always a multiplicative mask on `body`. Use it for non-reduction sparsity and bias-on-output patterns. Inside a `sum` it folds into the reduce's domain via mask-extraction; outside it stays as a `Cond` tag.

**Predicates** are conjunctions of affine inequalities over dim names: `<=`, `<`, `>=`, `>`, `==`, plus `+`, `-`, and `int * dim`. Cross-dim predicates (`nctx <= qctx`) are first-class — they ride along in the reduce's domain and survive interval merging.

## Kernel primitives

The kernel side mirrors a small slice of JAX/Numpy. Today's main backend is `stile.jax` (`tjax`); a numpy backend exists for prototyping.

```python
from stile import dim
import stile.jax as tjax

# Dims live in a global registry so specs and kernels share names.
qctx, nctx, dhead = dim('qctx', 128), dim('nctx', 512), dim('dhead', 16)

# Wrap concrete arrays with their dim signature.
Q = tjax.random.normal(key, qctx, dhead)        # has ShapeType (qctx, dhead)

# Slice. Result's st remembers it's [iqctx, iqctx+T).
q_tile = Q.slice(qctx, iqctx, iqctx + 32)

# Einsum. Both shapes and the AST track the contraction.
qk = tjax.einsum(Q, K, "qctx dhead, nctx dhead -> qctx nctx")

# Reductions. Either a method or via einsum.
m = qk.max(nctx)
s = qk.sum(nctx)

# Unary functions on TypedJaxArrays.
e = tjax.exp(qk - m.repeat(nctx))

# Multiplicative mask sugar — score * Cond(P, 1, 0).
masked = score.where("nctx <= qctx")

# Tagged-constant tensor — picks 0/1, 0/-inf, etc.
mult_mask = tjax.mask(score.type.st, "nctx <= qctx")              # 1 inside / 0 outside
bias_mask = tjax.mask(score.type.st, "nctx <= qctx", 0.0, -jnp.inf)  # 0 / -inf

# Rolled loops. Concrete bounds unroll; symbolic bounds emit a parametric reduce.
total = tjax.fori_loop(0, n, lambda i, acc: acc + body(i), init_val=0.0)

# Verify against a spec.
result = tjax.TypedResult(SPEC)
result.assign(o)                          # full-coverage type check
o.assert_equivalent(SPEC, nctx[:K])       # per-tile check with a slice override
result.done()                             # tile-coverage check (no gaps/overlaps)
```

## Status

Working today, with full structural verification:
- **Backends**: `stile.jax` (primary), `stile.numpy` (prototype).
- **Verified kernels**: matmul, online softmax, full flash attention, **tile-walking causal flash attention** (online softmax with bias-mask; structurally proven equivalent to a one-line `softmax[k where k<=q]` spec).
- **Spec features**: einsums, slices, reductions (`sum`, `max`, `softmax`), unary (`exp`, `sin`, `cos`, `sqrt`), multiplicative `where`-clauses, iteration-restricted `[d where P]` annotations, affine predicates with cross-dim references.
- **Kernel features**: slicing, einsum, all the unary/binary ops, `repeat`, `rearrange`, `fori_loop` (concrete-unroll path; symbolic-loop path with parametric reductions), `mask` intrinsic and `.where(...)` sugar.

In progress / future:
- **TypedPallas**: same type discipline, lowering to Pallas for actual GPU/TPU codegen.
- **TypedTorch**: a Torch backend exists but lags the JAX one.

## Why a type system?

Way back in the 1950s, before high-level languages, programs took big arrays of bytes as input and outputted other arrays of bytes. It was up to the programmer to remember in his head which bytes corresponded
to which semantic piece of the program. The invention of type annotations, structures, etc., was a step change in programming productivity because it gave semantic meanings to specific regions of memory.

The current state of the art for numerical programs is not unlike the 1950s mode of programming. We take big multidimensional arrays of floats, and output big multidimensional arrays of floats. The dimensions
are not semantically enforced to be different, and it's up to the programmer to remember the order of dimensions at all times. Mixing them is all too easy. Then assuming a program actually completed, you 
have another big array of floats, and you have no idea if it's right or not. If it doesn't give you the expected result, debugging a numerical program is a massive time-sink. The solution is therefore to
add guardrails to prevent making stupid mistakes, in other words a type system. 


## Running

```bash
uv run pytest tests/
```

Backend extras:

```bash
uv pip install -e ".[jax]"                 # JAX backend
uv pip install -e ".[torch]"               # Torch backend (lagging)
```
