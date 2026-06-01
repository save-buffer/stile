"""
FP-rounded evaluator over the raw `ExprType` tree.

Operates on the **un-normalized** ET so the kernel's actual operation
order is preserved — `(a + b) + c` and `a + (b + c)` are different
computations under floating-point and the evaluator must see them as
written. (Normalizing first would fold them to the same canonical form
and lose the per-tree-shape rounding profile we're trying to bound.)

Each AST-equivalent op attaches a single rounding-noise symbol with
coefficient `ε · |value|`, where `ε` is determined by the
`HardwareNumerics`. Leaf tensors look up their `AffineForm` in a
caller-provided dict; two evaluations of different trees that
reference the same leaf share the noise, so subtracting their
`AffineForm`s cancels everything except the per-op rounding gap.

Reduction order is read from the `HardwareNumerics`:
  - `"sequential"`: `N - 1` adds, partial sum grows linearly. Worst
    case under round-to-nearest. The default.
  - `"tree"`: log₂(N)-depth pairwise tree. Same op count but partial
    sums stay bounded — error scales with `log₂ N · ε`.
  - `"hw_fused"`: a single rounding step per `native_tile_size`-wide
    chunk, then those chunks combine via the cross-tile order (which
    we keep as `"sequential"` for now — refinement is on the todo).
"""
from __future__ import annotations

import math

from .affine import (
    AffineForm, fresh_noise,
    add, sub, scale, mul, div, exp, sqrt, reciprocal, maximum,
    affine_unary, round_fp, MACHINE_EPS,
)
from .hardware import HardwareNumerics, WORST_CASE
from ..type import (
    Constant, Tensor, UnaryOp, BinaryOp, Repeat, Reduce,
    Gather, Scatter, TagCond, dim_size, as_int,
)


def evaluate(
    expr,
    leaf_forms : "dict[str, AffineForm]",
    *,
    hardware : HardwareNumerics = WORST_CASE,
    tag_noise_cache : "dict | None" = None,
) -> AffineForm:
    """
    Evaluate an `ExprType` under the given `HardwareNumerics`, returning
    an `AffineForm` whose range bounds the kernel's floating-point
    output. `leaf_forms` maps tensor name → its starting AffineForm
    (typically a central value + a noise symbol bounded by its input
    range).

    `tag_noise_cache` is an optional caller-supplied dict mapping
    `(tag.domain, if_true, if_false)` → mask ε_noise. Passing the
    same dict to two evaluations ensures references to the same
    mask in both ETs share the same ε, so the subtraction in
    `exprs_close` cancels the mask term. Defaults to a fresh dict
    (mask correlations preserved only within a single evaluation).
    """
    if tag_noise_cache is None:
        tag_noise_cache = {}
    return _eval(
        expr, leaves=leaf_forms, hw=hardware,
        tag_noise_cache=tag_noise_cache,
    )


def _eval(node, leaves, hw : HardwareNumerics, tag_noise_cache : dict) -> AffineForm:
    if isinstance(node, Constant):
        return AffineForm.constant(float(node.value))

    if isinstance(node, Tensor):
        if node.tag is None:
            try:
                return leaves[node.name]
            except KeyError:
                raise KeyError(
                    f"AA evaluator: tensor `{node.name}` not in "
                    f"leaf_forms (have {sorted(leaves)})."
                ) from None
        # Tagged tensor (a TagCond mask or piecewise-constant). The
        # cell value at each position is one of the tag's branches.
        return _eval_tagged_tensor(node, leaves, hw, tag_noise_cache)

    if isinstance(node, UnaryOp):
        # Bias-mask special case: `exp(child + bias_mask)` where
        # bias_mask = Cond(P, 0, -inf). After exp:
        #   - P-positions: exp(child)  (range bounded by exp ∘ child).
        #   - ¬P positions: exp(-inf) = 0.
        # Treat as a piecewise-constant mid + tag noise bounded by
        # [0, max(exp(child))]. Without this shortcut the bias mask
        # propagates a `-inf` through AA's range formula and breaks.
        if node.op == "exp" and _is_bias_masked_add(node.child):
            return _eval_exp_of_bias_masked(
                node.child, leaves, hw, tag_noise_cache,
            )
        child = _eval(node.child, leaves, hw, tag_noise_cache)
        op = node.op
        if op == "exp":
            return round_fp(exp(child), hw.default_eps, label="exp-round")
        if op == "sqrt":
            return round_fp(sqrt(child), hw.default_eps, label="sqrt-round")
        if op == "sin":
            return round_fp(
                affine_unary(child, math.sin, math.cos, "sin-lin"),
                hw.default_eps, label="sin-round",
            )
        if op == "cos":
            return round_fp(
                affine_unary(child, math.cos, lambda x: -math.sin(x), "cos-lin"),
                hw.default_eps, label="cos-round",
            )
        raise NotImplementedError(f"AA evaluator: UnaryOp {op!r}")

    if isinstance(node, BinaryOp):
        lhs = _eval(node.lhs, leaves, hw, tag_noise_cache)
        rhs = _eval(node.rhs, leaves, hw, tag_noise_cache)
        op = node.op
        if op == "+":
            return round_fp(add(lhs, rhs), hw.default_eps, label="add-round")
        if op == "-":
            return round_fp(sub(lhs, rhs), hw.default_eps, label="sub-round")
        if op == "*":
            return round_fp(mul(lhs, rhs), hw.default_eps, label="mul-round")
        if op == "/":
            return round_fp(div(lhs, rhs), hw.default_eps, label="div-round")
        if op == "max":
            # `max` itself doesn't round (it's a comparison + select).
            return maximum(lhs, rhs)
        raise NotImplementedError(f"AA evaluator: BinaryOp {op!r}")

    if isinstance(node, Repeat):
        # Broadcasting: per-cell value unchanged; no extra rounding.
        return _eval(node.child, leaves, hw, tag_noise_cache)

    if isinstance(node, Reduce):
        child = _eval(node.child, leaves, hw, tag_noise_cache)
        n = as_int(dim_size(node.dim))
        if n is None:
            raise NotImplementedError(
                f"AA evaluator: reduce over non-concrete dim "
                f"{node.dim!r} not yet handled."
            )
        if node.op == "sum":
            return _sum_reduction(child, n, hw)
        if node.op == "max":
            # Max of N copies of a single AffineForm is the form
            # itself (max is non-arithmetic — under round-to-nearest
            # the per-cell rounding doesn't widen the post-reduce
            # range when cells happen to be equal). Looser bound is
            # acceptable; refining for non-equal cells is future work.
            return child
        raise NotImplementedError(f"AA evaluator: Reduce op {node.op!r}")

    if isinstance(node, Gather):
        # Output cell = source cell at idx position. AA-wise: each
        # output position can be any of the source's positions, so
        # the output value's bound = source's enclosing range. We
        # return the source's AffineForm directly — same noise
        # symbols, so subsequent ops still cancel correlations across
        # two gathers of the same source.
        return _eval(node.source, leaves, hw, tag_noise_cache)

    if isinstance(node, Scatter):
        # Output cells either get a source value or stay zero. Bound
        # is the union of source's range with {0}. Use a tag-style
        # noise to widen the form conservatively.
        src = _eval(node.source, leaves, hw, tag_noise_cache)
        lo, hi = src.range()
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        mid = 0.5 * (lo + hi)
        rad = 0.5 * (hi - lo)
        if rad == 0.0:
            return AffineForm.constant(mid)
        return AffineForm.with_noise(
            mid, fresh_noise("scatter-bound"), rad,
        )

    raise NotImplementedError(
        f"AA evaluator: unhandled ExprType {type(node).__name__}"
    )


def _eval_tagged_tensor(
    node, leaves, hw, tag_noise_cache : dict,
) -> AffineForm:
    """
    A tagged tensor `Tensor(tag=TagCond(P, if_true, if_false))` has
    a position-dependent cell value: either `if_true` (where P) or
    `if_false` (where ¬P). The piecewise-noise representation:

        AffineForm(mid, ((ε_mask, half_spread),))

    where `mid = (if_true + if_false) / 2`, `half_spread = |if_true -
    if_false| / 2`, and `ε_mask ∈ [-1, 1]` selects the branch (= +1
    chooses `if_true`, = -1 chooses `if_false`). The same `ε_mask`
    is reused for every reference to the same tag (cached by
    (domain, if_true, if_false)) so correlations across uses of the
    same mask cancel — `mask - mask = 0` exactly.

    Bias masks (`Cond(P, 0, -inf)` or similar) can't be represented
    in this finite form because `half_spread = inf`. The caller is
    expected to recognize `exp(... + bias_mask)` and route through
    `_eval_exp_of_bias_masked`. A bare encounter raises.
    """
    tag = node.tag
    if not (isinstance(tag.if_true, Constant)
            and isinstance(tag.if_false, Constant)):
        raise NotImplementedError(
            f"AA evaluator: tagged tensor `{node.name}` with "
            f"non-Constant branches isn't yet supported (saw "
            f"{type(tag.if_true).__name__}, "
            f"{type(tag.if_false).__name__})."
        )
    t = float(tag.if_true.value)
    f = float(tag.if_false.value)
    if not (math.isfinite(t) and math.isfinite(f)):
        raise NotImplementedError(
            f"AA evaluator: bias mask `{node.name}` has a non-finite "
            f"branch (if_true={t}, if_false={f}). Bias masks are only "
            f"AA-meaningful inside an `exp(... + mask)` wrapper, which "
            f"the evaluator recognizes and short-circuits. Wrap the "
            f"add-then-exp at the call site."
        )
    mid = 0.5 * (t + f)
    half_spread = 0.5 * abs(t - f)
    if half_spread == 0.0:
        return AffineForm.constant(mid)
    key = (tag.domain, t, f)
    if key not in tag_noise_cache:
        tag_noise_cache[key] = fresh_noise(f"mask-{node.name}")
    return AffineForm.with_noise(mid, tag_noise_cache[key], half_spread)


def _is_bias_masked_add(node) -> bool:
    """
    True iff `node` is `BinaryOp(+, x, mask)` where `mask` is a
    Tensor with a bias-form TagCond tag (one branch is ±inf). The
    `exp(...)` handler peeks for this so it can short-circuit the
    bound — bias masks aren't representable in AA on their own.
    """
    if not isinstance(node, BinaryOp) or node.op != "+":
        return False
    for side in (node.lhs, node.rhs):
        if (isinstance(side, Tensor) and isinstance(side.tag, TagCond)
                and isinstance(side.tag.if_true, Constant)
                and isinstance(side.tag.if_false, Constant)):
            t = float(side.tag.if_true.value)
            f = float(side.tag.if_false.value)
            if not (math.isfinite(t) and math.isfinite(f)):
                return True
    return False


def _eval_exp_of_bias_masked(
    add_node, leaves, hw, tag_noise_cache : dict,
) -> AffineForm:
    """
    `exp(x + Cond(P, 0, -inf))` evaluates as:
      - P-positions:  exp(x + 0) = exp(x)
      - ¬P positions: exp(x + (-inf)) = exp(-inf) = 0.
    Bound = enclosing range of {0} ∪ exp(x.range()). Represent as a
    piecewise-noise AffineForm with the mask's own shared ε so
    subsequent uses correlate. The on-branch value is bounded by the
    fully-rounded exp(x); the off-branch is identically zero.
    """
    # Find x and the mask among the +'s operands.
    if isinstance(add_node.lhs, Tensor) and isinstance(add_node.lhs.tag, TagCond):
        mask, x = add_node.lhs, add_node.rhs
    else:
        mask, x = add_node.rhs, add_node.lhs
    t = float(mask.tag.if_true.value)
    f = float(mask.tag.if_false.value)
    # Determine which branch is the "on" (finite, typically 0) and
    # which is "off" (-inf).
    if math.isfinite(t) and not math.isfinite(f):
        on_branch_finite = t
    elif math.isfinite(f) and not math.isfinite(t):
        on_branch_finite = f
    else:
        raise ValueError(
            "bias-mask add: expected exactly one finite + one -inf branch"
        )
    # Compute exp(x + on_branch_finite) under fp rounding (typically
    # on_branch_finite = 0 so this is just exp(x)).
    x_form = _eval(x, leaves, hw, tag_noise_cache)
    if on_branch_finite != 0.0:
        x_form = round_fp(
            add(x_form, AffineForm.constant(on_branch_finite)),
            hw.default_eps, label="bias-add-round",
        )
    on_value = round_fp(exp(x_form), hw.default_eps, label="exp-round")
    # Output: union of {0} and on_value.range(). Piecewise-noise form
    # with the SAME mask noise so two exp(... + same_mask) instances
    # correlate.
    lo, hi = on_value.range()
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    mid = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)
    if rad == 0.0:
        return AffineForm.constant(mid)
    key = ("bias-mask-exp", mask.tag.domain, on_branch_finite)
    if key not in tag_noise_cache:
        tag_noise_cache[key] = fresh_noise(f"bias-mask-exp-{mask.name}")
    return AffineForm.with_noise(mid, tag_noise_cache[key], rad)


def _sum_reduction(
    child : AffineForm, n : int, hw : HardwareNumerics,
) -> AffineForm:
    """
    Sum of N copies of `child` under `hw.reduction_order`. The
    accumulator runs at `hw.acc_dtype`, which may be wider than the
    op's input dtype (e.g. bf16 inputs, fp32 accumulator on TPU).
    """
    if n <= 0:
        return AffineForm.constant(0.0)
    if n == 1:
        return child
    acc_eps = hw.acc_eps

    if hw.reduction_order == "sequential":
        # N-1 additions; partial sum after k adds has magnitude
        # ≤ k · max|child|; each add contributes ≤ acc_eps · |partial|.
        # Sum the bound: total error ≤ acc_eps · max|child| · Σ k
        # for k=1..N-1 = acc_eps · max|child| · N(N-1)/2.
        scaled = scale(child, float(n))
        lo, hi = child.range()
        max_mag = max(abs(lo), abs(hi))
        error = acc_eps * max_mag * (n * (n - 1) // 2)
        if error == 0.0:
            return scaled
        return add(
            scaled,
            AffineForm.with_noise(
                0.0, fresh_noise(f"seq-sum-{n}"), error,
            ),
        )

    if hw.reduction_order == "tree":
        # Pairwise tree: log₂(N) depth. At each level, partial sums
        # have magnitude ≤ 2 · prev_magnitude. Total error per leaf
        # contributes through log₂(N) ops, each at acc_eps. Total:
        # acc_eps · max|child| · N · log₂(N).
        scaled = scale(child, float(n))
        lo, hi = child.range()
        max_mag = max(abs(lo), abs(hi))
        depth = max(1, math.ceil(math.log2(n)))
        error = acc_eps * max_mag * n * depth
        if error == 0.0:
            return scaled
        return add(
            scaled,
            AffineForm.with_noise(
                0.0, fresh_noise(f"tree-sum-{n}"), error,
            ),
        )

    if hw.reduction_order == "hw_fused":
        # Each `native_tile_size`-wide chunk reduces in one rounded
        # step; chunks combine via sequential cross-tile order.
        tile = max(1, hw.native_tile_size)
        n_tiles = math.ceil(n / tile)
        scaled = scale(child, float(n))
        lo, hi = child.range()
        max_mag = max(abs(lo), abs(hi))
        # Within-tile error: 1 rounding per tile, each at magnitude
        # acc_eps · (tile · max|child|).
        within_tile_error = n_tiles * acc_eps * (tile * max_mag)
        # Cross-tile sequential combine: n_tiles - 1 adds, each at
        # magnitude up to (n_tiles · tile · max|child|).
        cross_tile_error = (
            acc_eps * (n_tiles * tile * max_mag)
            * (n_tiles * (n_tiles - 1) // 2)
        )
        error = within_tile_error + cross_tile_error
        if error == 0.0:
            return scaled
        return add(
            scaled,
            AffineForm.with_noise(
                0.0,
                fresh_noise(f"hwfused-sum-{n}-tile{tile}"),
                error,
            ),
        )

    raise ValueError(
        f"unknown reduction_order {hw.reduction_order!r}"
    )


def rounding_error_bound(
    form : AffineForm, leaf_forms : "dict[str, AffineForm]",
) -> float:
    """
    Sum of `|coeff|` over noise symbols in `form` that are *not* part
    of any leaf form — i.e. the per-op rounding noises the evaluator
    introduced, separate from the input-uncertainty noises that came
    in via `leaf_forms`. Useful for "how much numerical error did
    this kernel pick up purely from finite precision, assuming exact
    inputs?".
    """
    leaf_symbol_ids = set()
    for leaf in leaf_forms.values():
        for s, _ in leaf.noise:
            leaf_symbol_ids.add(s.id)
    return sum(
        abs(c) for s, c in form.noise if s.id not in leaf_symbol_ids
    )


def exprs_close(
    et_a, et_b,
    *,
    leaf_forms : "dict[str, AffineForm]",
    hardware : HardwareNumerics = WORST_CASE,
    tolerance : float = 0.0,
) -> "tuple[bool, AffineForm]":
    """
    Evaluate both ETs under `hardware` with shared `leaf_forms`, then
    return `(close, diff)` where `diff = a - b` and `close` is True
    iff `diff.range() ⊆ [-tolerance, tolerance]`.

    `diff`'s noise terms attribute the gap to specific rounding
    sources — handy for "where did the error come from" diagnostics.
    """
    # Shared cache across both evaluations: a mask referenced from
    # both ETs gets the same ε_noise, so its term cancels in the
    # difference. (Without sharing, two `Cond(P, 1, 0)` references
    # from different evaluations would generate independent ε's and
    # the mask wouldn't cancel.)
    shared_tag_cache : dict = {}
    a = evaluate(
        et_a, leaf_forms, hardware=hardware,
        tag_noise_cache=shared_tag_cache,
    )
    b = evaluate(
        et_b, leaf_forms, hardware=hardware,
        tag_noise_cache=shared_tag_cache,
    )
    diff = sub(a, b)
    lo, hi = diff.range()
    close = (-tolerance <= lo) and (hi <= tolerance)
    return close, diff
