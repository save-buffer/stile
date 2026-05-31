"""
Core `stile.type` algebra: `type_from_binary_op` broadcasting, the
`neg` helper, and `Sliced` bound validation. Backend-agnostic — these
operate directly on `Type` / `ExprType`, the layer every frontend
(jax / torch / numpy / triton, and external DSLs) builds on.
"""
import pytest

from stile import dim
import stile.type as st
from stile.type import Type, Tensor, Constant, Sliced, type_from_binary_op, neg
from stile.indexing import SymbolicInt, to_affine


# --- type_from_binary_op: scalar / Constant broadcasting ---------------

def test_scalar_constant_broadcasts_against_tensor(reset):
    """A scalar-shaped operand (`st == ()`) — e.g. a literal `2` — must
    broadcast against any tensor shape. Triton-style kernels do `x * 2`,
    `max(x, 0)` constantly."""
    N = dim("N", 8)
    x = Type(st=(N[0:1],), et=Tensor((N,), name="X"))
    two = Type(st=(), et=Constant(2.0))
    out = type_from_binary_op(x, two, "*")
    assert out.st == (N[0:1],)        # keeps the tensor's shape
    assert out.et == st.BinaryOp("*", x.et, two.et)


def test_scalar_constant_broadcasts_either_side(reset):
    """Broadcast works with the scalar on the left, too."""
    N = dim("N", 8)
    x = Type(st=(N,), et=Tensor((N,), name="X"))
    zero = Type(st=(), et=Constant(0.0))
    out = type_from_binary_op(zero, x, "-")
    assert out.st == (N,)


def test_nonscalar_shape_mismatch_still_rejected(reset):
    """Two genuinely-different non-scalar shapes still raise — the
    broadcast only relaxes the scalar case."""
    M = dim("M", 4)
    N = dim("N", 8)
    a = Type(st=(M,), et=Tensor((M,), name="A"))
    b = Type(st=(N,), et=Tensor((N,), name="B"))
    with pytest.raises(ValueError, match="same shapes"):
        type_from_binary_op(a, b, "+")


def test_scalar_broadcast_keeps_tensor_dtype(reset):
    """The dtype-bearing (tensor) side's dtype survives the broadcast;
    the scalar side is dtype-less."""
    N = dim("N", 8)
    x = Type(st=(N,), et=Tensor((N,), name="X"), dt=st.DataType.float32)
    two = Type(st=(), et=Constant(2.0))
    out = type_from_binary_op(x, two, "*")
    assert out.dt == st.DataType.float32


# --- neg ----------------------------------------------------------------

def test_neg_keeps_shape_and_dtype(reset):
    """`neg(x)` is `0 - x`, preserving x's shape and dtype."""
    N = dim("N", 8)
    x = Type(st=(N,), et=Tensor((N,), name="X"), dt=st.DataType.float32)
    out = neg(x)
    assert out.st == (N,)
    assert out.dt == st.DataType.float32
    assert out.et == st.BinaryOp("-", Constant(0.0), x.et)


# --- Sliced bound validation at construction ----------------------------

def test_sliced_accepts_valid_bound_types(reset):
    """int / AffineExpr / SymbolicInt bounds are all accepted."""
    N = dim("N", 8)
    Sliced(N, 0, 4)                         # int
    Sliced(N, to_affine(0), to_affine(4))   # AffineExpr
    k = SymbolicInt("k")
    Sliced(N, k, to_affine(k) + 1)          # SymbolicInt + AffineExpr


def test_sliced_rejects_foreign_bound_with_clear_error(reset):
    """A non-SymbolicIndex bound (e.g. a frontend's own tracer object)
    raises at construction, naming the offending bound — rather than
    blowing up later inside an unrelated `__eq__`."""
    N = dim("N", 8)

    class FakeTracer:
        def __eq__(self, other):
            raise TypeError("Cannot use a traced value in a boolean context")

    with pytest.raises(TypeError, match="Sliced end bound must be a SymbolicIndex"):
        Sliced(N, 0, FakeTracer())
    with pytest.raises(TypeError, match="Sliced start bound must be a SymbolicIndex"):
        Sliced(N, FakeTracer(), 4)
