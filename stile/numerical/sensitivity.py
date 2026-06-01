"""
Dtype sensitivity analysis. Re-runs a typed kernel under a swapped
precision plan and reports how much the AA rounding bound widens —
in total and per op.

Use case: ablation. "Which input tensor (or which op) drives the
numerical error if I downgrade it from fp32 to fp8?" The typed
kernel's named inputs (e.g. `A = tjax.tensor(..., name="A")`) get
cast to the target dtype, the kernel re-runs, and the per-op AA
noise mass is compared between runs. Labels like `"add-round"`,
`"einsum-mul-round"`, `"exp-round"` come straight from the noise
symbols that the eager-AA pipeline attaches at each op.

Why a comparison and not absolute bounds: AA is sound but loose by
orders of magnitude on random data (the `mul` cross-term dominates,
random concentration is way tighter than worst-case). The looseness
is data-dependent, not dtype-dependent — so it cancels when the same
kernel is run twice and only the dtypes differ. The widening ratio
is the part of the bound that's actually attributable to precision
choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


_REDUCTION_PREFIXES = ("seq-sum-", "tree-sum-", "hwfused-sum-")


def _is_rounding_label(label : str) -> bool:
    """
    A noise label is "rounding-like" — i.e. scales with the
    operating dtype — iff it ends in `-round` (per-op multiplier /
    add / exp / etc. rounding) or starts with a reduction prefix
    (`seq-sum-N`, `tree-sum-N`, `hwfused-sum-N-tileK` — accumulator
    rounding). Linearization-noise labels (`mul-cross`, `*-lin`,
    `max-bound`, mask/scatter bounds) are excluded — those are AA's
    worst-case conservatism, not floating-point rounding.
    """
    if label.endswith("-round"):
        return True
    return any(label.startswith(p) for p in _REDUCTION_PREFIXES)


@dataclass(frozen=True)
class Sensitivity:
    """
    Output of `sensitivity_analysis`. Holds total + per-op AA bound
    widening for a single dtype-swap experiment.

    Fields:
      - `ref_bound`: AA rounding-noise mass under the reference dtypes.
      - `low_bound`: same mass under the swapped dtypes.
      - `per_label`: per-noise-label breakdown, label → `(ref, low,
        ratio)`. Labels match the strings each op attaches when calling
        `round_fp(..., label=...)` — useful for picking out "which op
        is responsible" for the widening.
    """
    ref_bound : float
    low_bound : float
    per_label : "dict[str, tuple[float, float, float]]" = field(default_factory=dict)

    @property
    def widening(self) -> float:
        """
        `low_bound / ref_bound` over the full non-leaf noise mass.

        Caveat: includes data-dependent AA linearization noise (e.g.
        `mul-cross` from the AA multiplication primitive, `exp-lin`
        from nonlinear unary linearizations). Those don't depend on
        dtype, so they cancel in the *ratio* — but they often
        DOMINATE the absolute magnitude, so this total ends up reading
        ≈ 1.0 even when the dtype-rounding noises widen by orders of
        magnitude. For precision-ablation use cases, prefer
        `rounding_widening` instead — that one restricts to the
        dtype-driven labels and gives an interpretable headline number.
        """
        if self.ref_bound == 0.0:
            return float("inf") if self.low_bound > 0.0 else 1.0
        return self.low_bound / self.ref_bound

    @property
    def rounding_widening(self) -> float:
        """
        `low / ref` restricted to *dtype-rounding* noise labels —
        the labels that genuinely move with precision (`*-round` from
        per-op rounding, `seq-sum-N` / `tree-sum-N` / `hwfused-sum-*`
        from accumulator rounding). Excludes AA-linearization noises
        (`mul-cross`, `*-lin`, `max-bound`, mask/scatter bounds) which
        are data-dependent and cancel exactly in the ratio.

        This is the right headline number for "how much worse does
        rounding get under this precision swap?" — it cleanly isolates
        the dtype effect from AA's worst-case linearization
        conservatism.
        """
        ref_total = 0.0
        low_total = 0.0
        for label, (ref, low, _) in self.per_label.items():
            if _is_rounding_label(label):
                ref_total += ref
                low_total += low
        if ref_total == 0.0:
            return float("inf") if low_total > 0.0 else 1.0
        return low_total / ref_total

    def top(self, n : int = 5) -> "list[tuple[str, float, float, float]]":
        """
        Top-`n` labels by widening ratio. Each row is `(label, ref,
        low, ratio)`. Labels with `ref == 0` (the swap introduced a new
        noise source) sort to the top with infinite ratio.
        """
        rows = [
            (label, ref, low, ratio)
            for label, (ref, low, ratio) in self.per_label.items()
        ]
        rows.sort(key=lambda r: (-r[3] if r[3] != float("inf") else -1e308))
        return rows[:n]

    def summary(self, n : int = 5) -> str:
        """
        Multi-line summary suitable for `print()`. Headline number
        is `rounding_widening` — the dtype-driven part of the bound —
        which is the one that actually moves with precision choice.
        Total widening (including data-dependent linearization noise)
        and the top-`n` per-label rows follow.
        """
        rw = self.rounding_widening
        rw_s = "inf" if rw == float("inf") else f"{rw:.2f}×"
        lines = [
            f"rounding widening: {rw_s}  (the dtype-driven part)",
            f"total widening:   {self.widening:.2f}×  "
            f"(ref={self.ref_bound:.4e}, low={self.low_bound:.4e})",
            "top-contributing labels:",
        ]
        for label, ref, low, ratio in self.top(n):
            ratio_s = "inf" if ratio == float("inf") else f"{ratio:6.2f}×"
            lines.append(
                f"  {ratio_s}  {label:<24s} "
                f"(ref={ref:.4e}, low={low:.4e})"
            )
        return "\n".join(lines)


def sensitivity_analysis(
    fn : Callable, args : tuple, *, swap : "dict[str, str]",
) -> Sensitivity:
    """
    Run `fn(*args)` twice — once with the original typed inputs, once
    with named inputs cast to the dtypes in `swap` — and return a
    `Sensitivity` summarizing how much the AA rounding bound widens.

    Identification of which arg to swap is by name: each typed input
    is expected to have `arg.type.et.name` set (the `name="A"` kwarg on
    `tjax.tensor` / `ttorch.tensor` / `tnumpy.tensor`). Args whose name
    doesn't match anything in `swap` are passed through unchanged.

    The cast uses the typed value's `.astype(dtype_str)` method (which
    each backend implements via its own `astype` / `to`).

    The single-output assumption: `fn` must return one typed value (or
    something with `.aa`). For multi-output kernels, wrap with a
    `lambda *args: my_kernel(*args)[0]` to pick the output to analyze.
    """
    ref_out = fn(*args)
    new_args = tuple(_apply_swap(a, swap) for a in args)
    low_out = fn(*new_args)
    return _diff(args, ref_out, new_args, low_out)


def _apply_swap(arg, swap : "dict[str, str]"):
    name = _name_of(arg)
    if name is None or name not in swap:
        return arg
    return arg.astype(swap[name])


def _name_of(arg) -> "str | None":
    """
    Best-effort: pull `.type.et.name` off the arg. Used to match
    args against the `swap` dict's keys.
    """
    et = getattr(getattr(arg, "type", None), "et", None)
    return getattr(et, "name", None) if et is not None else None


def _leaf_ids(args) -> "set[int]":
    """
    Collect the noise-symbol IDs from each arg's leaf AA. These are
    the "input range" noises — we exclude them from the rounding-noise
    mass so the widening reflects what the ops contributed, not the
    inputs themselves.
    """
    ids : set[int] = set()
    for a in args:
        aa = getattr(a, "aa", None)
        if aa is None:
            continue
        for sym, _ in aa.noise:
            ids.add(sym.id)
    return ids


def _per_label_mass(aa, leaf_ids : "set[int]") -> "dict[str, float]":
    """
    Group an AA's non-leaf noise terms by label and sum |coeff|
    within each group.
    """
    by_label : dict[str, float] = {}
    if aa is None:
        return by_label
    for sym, c in aa.noise:
        if sym.id in leaf_ids:
            continue
        by_label[sym.label] = by_label.get(sym.label, 0.0) + abs(c)
    return by_label


def _diff(ref_args, ref_out, low_args, low_out) -> Sensitivity:
    ref_leaves = _leaf_ids(ref_args)
    low_leaves = _leaf_ids(low_args)
    ref_aa = getattr(ref_out, "aa", None)
    low_aa = getattr(low_out, "aa", None)
    ref_by_label = _per_label_mass(ref_aa, ref_leaves)
    low_by_label = _per_label_mass(low_aa, low_leaves)

    all_labels = set(ref_by_label) | set(low_by_label)
    per_label : dict[str, tuple[float, float, float]] = {}
    for label in all_labels:
        r = ref_by_label.get(label, 0.0)
        l = low_by_label.get(label, 0.0)
        if r > 0:
            ratio = l / r
        elif l > 0:
            ratio = float("inf")
        else:
            ratio = 1.0
        per_label[label] = (r, l, ratio)

    return Sensitivity(
        ref_bound=sum(ref_by_label.values()),
        low_bound=sum(low_by_label.values()),
        per_label=per_label,
    )
