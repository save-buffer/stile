"""
`[d where P]` dim-annotation syntax: a predicate on the *iteration* of a
reduction. Lowers differently per op:
  - `sum[d where P](body)`     → mult-mask `body * Cond(P, 1, 0)`
  - `max[d where P](body)`     → bias-mask `body + Cond(P, 0, -inf)`
  - `softmax[d where P](body)` → bias-mask before macro expansion

The verifier sees both the mult and bias forms via #7+#10+#11 and converges
them, so the parser just picks the encoding that's mathematically correct
for the surrounding op.
"""
import pytest

from stile import dim, reset_stile
from stile.specification import parse_spec_into_type
from stile.verification import (
    normalize, verify_exprs_equivalent, NormalizedReduce,
)
from stile.indexing import LoopVariable


@pytest.fixture
def reset():
    yield
    reset_stile()


def _single_factor(expr):
    n = normalize(expr)
    assert len(n.num.factors) == 1
    return next(iter(n.num.factors))


def test_sum_dim_annotation_predicate_equivalent_to_where_clause(reset):
    """`sum[d where P](body)` ≡ `(sum[d](body) where P)` (the older clause
    form). Both lower to a mult-mask folded into the reduce's domain."""
    dim("DAN", 8)
    spec_annot = parse_spec_into_type("sum[DAN where DAN >= 4](DAN) -> ")
    spec_clause = parse_spec_into_type("sum[DAN](DAN where DAN >= 4) -> ")
    assert verify_exprs_equivalent(spec_annot.et, spec_clause.et)


def test_max_dim_annotation_predicate_uses_bias_form(reset):
    """`max[d where P](body)` lowers to `max[d](body + Cond(P, 0, -inf))`
    — bias-form is the right encoding for max because masked positions
    must vanish through the max identity (`-inf`), not the sum identity
    (`0`)."""
    dim("DAMN", 8)
    dim("DAMQ", 8)
    spec = parse_spec_into_type("max[DAMN where DAMN <= DAMQ](DAMN DAMQ) -> DAMQ")
    factor = _single_factor(spec.et)
    assert isinstance(factor, NormalizedReduce)
    assert factor.op == "max"
    # The reduce's domain has both `N` and `Q` as variables — the cross-
    # variable predicate landed in extras, exactly like the where-clause
    # path produces it.
    assert {LoopVariable("DAMN"), LoopVariable("DAMQ")} <= factor.domain.variables


def test_softmax_dim_annotation_predicate_restricts_iteration(reset):
    """`softmax[d where P](body)` restricts both numerator and
    denominator to P. Compare against the explicit-formula form to
    confirm convergence."""
    dim("DASMQ", 8)
    dim("DASMK", 8)
    spec_annot = parse_spec_into_type(
        "softmax[DASMK where DASMK <= DASMQ](DASMQ DASMK)"
    )
    # Explicit form (no softmax macro): mult-mask after exp, both num
    # and den restricted by the same predicate. Bias and mult forms
    # converge in normalization (#7), so the two specs must match.
    spec_explicit = parse_spec_into_type(
        "(exp(DASMQ DASMK) where DASMK <= DASMQ) / "
        "(sum[DASMK](exp(DASMQ DASMK) where DASMK <= DASMQ) -> DASMQ DASMK)"
    )
    assert verify_exprs_equivalent(spec_annot.et, spec_explicit.et)


def test_dim_annotation_predicate_validates_dim_in_shape(reset):
    """A predicate referencing a dim not in the body's shape is
    rejected at parse time."""
    dim("DAVN", 8)
    dim("DAVM", 4)
    with pytest.raises(ValueError, match="not in the body's shape"):
        parse_spec_into_type("sum[DAVN where DAVM >= 0](DAVN) -> ")


def test_dim_annotation_predicate_with_two_dims(reset):
    """Predicate over two body dims (the reduction dim and an outer
    one) — same shape as the causal-attention case."""
    dim("DAQ", 8)
    dim("DAK", 8)
    spec = parse_spec_into_type("sum[DAK where DAK <= DAQ](DAQ DAK) -> DAQ")
    factor = _single_factor(spec.et)
    assert isinstance(factor, NormalizedReduce)
    assert factor.op == "sum"
    # Domain carries: K range AND K <= Q.
    conj = next(iter(factor.domain.disjuncts))
    assert len(conj) == 3


def test_softmax_predicate_unrelated_to_where_outside(reset):
    """`softmax[d where P](body)` and `softmax[d](body) where P` mean
    different things. The first restricts the softmax's iteration; the
    second multiplies the softmax's *output* by a 0/1 mask."""
    dim("DAUQ", 8)
    dim("DAUK", 8)
    annot = parse_spec_into_type("softmax[DAUK where DAUK <= DAUQ](DAUQ DAUK)")
    output_mask = parse_spec_into_type("softmax[DAUK](DAUQ DAUK) where DAUK <= DAUQ")
    # These are semantically different surface forms; verifier should
    # NOT consider them equivalent.
    assert not verify_exprs_equivalent(annot.et, output_mask.et)
