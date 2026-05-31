try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch support requires the torch extra: pip install stile[torch]"
    ) from None

import stile.type as t
from ..type import *
from ..specification import parse_spec_into_type
from ..verification import verify_types_equivalent, verify_exprs_equivalent
from ..numerical import (
    AffineForm, leaf_aa_from_array, active_hardware,
    compose_binary, compose_unary, compose_einsum,
    dtype_name_of,
)

import einops


# Sentinel: `aa` defaults to "compute from tensor"; explicit `None` opts
# out. Distinguishes "caller passed None" from "caller didn't pass."
_NO_AA_DEFAULT = object()


# String-name → torch.dtype, for the `astype(dtype_str)` path used by
# `sensitivity_analysis`. Keys match `MACHINE_EPS` (so the same dtype
# string round-trips through eps lookup and the cast).
_TORCH_DTYPE_MAP = {
    "float64":  torch.float64,
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}
# FP8 variants exist on newer torch versions; add them when present so
# import doesn't break on older installs without them.
for _name, _attr in [
    ("fp8_e4m3", "float8_e4m3fn"),
    ("fp8_e5m2", "float8_e5m2"),
]:
    if hasattr(torch, _attr):
        _TORCH_DTYPE_MAP[_name] = getattr(torch, _attr)


# DataType -> torch.dtype, for building reference symbolic inputs in a
# declared dtype. Reuses the `_TORCH_DTYPE_MAP` string keys (which match
# `DataType.value`).
def _datatype_to_torch(dt : "DataType | None"):
    if dt is None:
        return torch.float32
    return _TORCH_DTYPE_MAP.get(dt.value, torch.float32)


# torch.dtype -> stile DataType (None for dtypes stile doesn't model).
_TORCH_TO_DATATYPE = {
    torch.bfloat16: DataType.bfloat16,
    torch.float32:  DataType.float32,
    torch.float64:  DataType.float64,
}

def dtype_to_datatype(torch_dtype) -> "DataType | None":
    """Map a `torch.dtype` to the stile `DataType` it corresponds to, or
    `None` if stile doesn't model it (so it acts as a dtype wildcard)."""
    return _TORCH_TO_DATATYPE.get(torch_dtype)


def make_symbolic_input(type : Type) -> "TypedTorchTensor":
    """Build a zero-backed `TypedTorchTensor` for `type` — a symbolic input
    for running a torch reference function. The array's dtype follows
    `type.dt` (default float32) so the reference's output dtype is
    meaningful; the values are irrelevant (we only read the result's type)."""
    shape = tuple(as_int(dim_size(d)) for d in type.st)
    arr = torch.zeros(shape, dtype=_datatype_to_torch(type.dt))
    return TypedTorchTensor(arr, type)


def _aa_of(value) -> "AffineForm | None":
    """Pull the `.aa` off a TypedTorchTensor, or wrap a numeric
    scalar as a zero-radius constant form."""
    if isinstance(value, TypedTorchTensor):
        return value.aa
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return AffineForm.constant(float(value))
    return None


class TypedTorchTensor:
    def __init__(
        self, tensor : torch.Tensor, type : Type,
        *, aa : "AffineForm | None" = _NO_AA_DEFAULT,
    ):
        # `aa` is the always-on affine-form bound on this value.
        # Defaults to a leaf form derived from `tensor.min()/max()` at
        # construction; per-op handlers (step 5) compose new AAs from
        # input AAs as values flow through typed ops.
        self.tensor = tensor
        self.type = type
        if aa is _NO_AA_DEFAULT:
            aa = leaf_aa_from_array(tensor)
        self.aa = aa

    @property
    def data_ptr(self) -> int:
        return self.tensor.data_ptr()

    def astype(self, dtype : str) -> "TypedTorchTensor":
        """Recast the underlying tensor to `dtype` (a `MACHINE_EPS`
        key such as `"bfloat16"` / `"fp8_e4m3"`); used by
        `sensitivity_analysis` to swap a named input's precision
        without changing shape or stile type."""
        torch_dtype = _TORCH_DTYPE_MAP[dtype]
        return TypedTorchTensor(self.tensor.to(torch_dtype), self.type)

    def slice(self, dim : FullDim, start : int, end : int) -> "TypedTorchTensor":
        slice_expr = []
        for i, d in enumerate(self.type.st):
            if dim_contains(d, dim):
                slice_expr.append(slice(start, end))
            else:
                slice_expr.append(slice(None))

        new_type = self.type.slice(dim, start, end)
        return TypedTorchTensor(self.tensor[tuple(slice_expr)], new_type)

    def repeat(self, dim : Dim) -> "TypedTorchTensor":
        nrepeats = dim_size(dim)
        repeated_tensor = self.tensor.unsqueeze(0).repeat(nrepeats, *([1] * self.tensor.dim()))
        new_type = self.type.repeat(dim)
        return TypedTorchTensor(repeated_tensor, new_type)

    def rearrange(self, *dims : Dim) -> "TypedTorchTensor":
        dims = tuple(dim_full_dim(d) for d in dims)

        dims_by_name = {}
        lhs_str = ""
        for d in self.type.st:
            name = dim_name(d)
            dims_by_name[name] = d
            lhs_str += f"{name} "

        rhs_str = ""
        names = [dim_name(d) for d in dims]
        for n in names:
            if n not in dims_by_name:
                raise ValueError(f"Trying to rearrange with unknown dim {n}")
            rhs_str += f"{n} "

        new_tensor = einops.rearrange(self.tensor, f"{lhs_str} -> {rhs_str}")
        new_type = self.type.rearrange(*dims)
        return TypedTorchTensor(new_tensor, new_type)

    def reduce(self, op : ReduceOpType, dim : Dim) -> "TypedTorchTensor":
        new_type = self.type.reduce(op, dim)

        for i, d in enumerate(self.type.st):
            if dim_name(dim) == dim_name(d):
                ireduction_dim = i
                break
        match op:
            case "sum":
                new_tensor = self.tensor.sum(dim=ireduction_dim)
            case "max":
                new_tensor = self.tensor.max(dim=ireduction_dim).values
        return TypedTorchTensor(new_tensor, new_type)

    def sum(self, dim : Dim) -> "TypedTorchTensor":
        return self.reduce("sum", dim)

    def max(self, dim : Dim) -> "TypedTorchTensor":
        return self.reduce("max", dim)

    def __add__(self, other) -> "TypedTorchTensor":
        return _binary_op_helper(self, other, "+")

    def __sub__(self, other) -> "TypedTorchTensor":
        return _binary_op_helper(self, other, "-")

    def __mul__(self, other) -> "TypedTorchTensor":
        return _binary_op_helper(self, other, "*")

    def __truediv__(self, other) -> "TypedTorchTensor":
        return _binary_op_helper(self, other, "/")

    def __radd__(self, other) -> "TypedTorchTensor":
        return _binary_op_helper(other, self, "+")

    def __rsub__(self, other) -> "TypedTorchTensor":
        return _binary_op_helper(other, self, "-")

    def __rmul__(self, other) -> "TypedTorchTensor":
        return _binary_op_helper(other, self, "*")

    def __rtruediv__(self, other) -> "TypedTorchTensor":
        return _binary_op_helper(other, self, "/")

    def __matmul__(self, other) -> "TypedTorchTensor":
        return einsum(self, other, "M N, N K -> M K")

    def assert_equivalent(self, spec : str, *dim_override : Dim):
        expected_type = parse_spec_into_type(spec)
        expected_type = override_dims_in_type(expected_type, *dim_override)
        are_equivalent = verify_exprs_equivalent(
            expected_type.et,
            self.type.et,
        )
        assert are_equivalent


def _binary_op_helper(
    slf : TypedTorchTensor | float,
    other : TypedTorchTensor | float,
    op : BinaryOpType,
) -> TypedTorchTensor:
    lhs_type = slf.type if isinstance(slf, TypedTorchTensor) else slf
    rhs_type = other.type if isinstance(other, TypedTorchTensor) else other
    new_type = type_from_binary_op(lhs_type, rhs_type, op)

    lhs = slf.tensor if isinstance(slf, TypedTorchTensor) else slf
    rhs = other.tensor if isinstance(other, TypedTorchTensor) else other

    # For operations that need tensors (like maximum), convert scalars
    if op == "max":
        if not isinstance(lhs, torch.Tensor):
            lhs = torch.tensor(lhs)
        if not isinstance(rhs, torch.Tensor):
            rhs = torch.tensor(rhs)

    match op:
        case "+":
            new_tensor = lhs + rhs
        case "-":
            new_tensor = lhs - rhs
        case "*":
            new_tensor = lhs * rhs
        case "/":
            new_tensor = lhs / rhs
        case "max":
            new_tensor = torch.maximum(lhs, rhs)
        case _:
            raise ValueError(f"Unknown op {op}")

    new_aa = compose_binary(
        op, _aa_of(slf), _aa_of(other), dtype_name_of(new_tensor),
    )
    return TypedTorchTensor(new_tensor, new_type, aa=new_aa)


def exp(x : TypedTorchTensor) -> TypedTorchTensor:
    new_tensor = torch.exp(x.tensor)
    return TypedTorchTensor(
        new_tensor, t.exp(x.type),
        aa=compose_unary("exp", x.aa, dtype_name_of(new_tensor)),
    )


def sin(x : TypedTorchTensor) -> TypedTorchTensor:
    new_tensor = torch.sin(x.tensor)
    return TypedTorchTensor(
        new_tensor, t.sin(x.type),
        aa=compose_unary("sin", x.aa, dtype_name_of(new_tensor)),
    )


def cos(x : TypedTorchTensor) -> TypedTorchTensor:
    new_tensor = torch.cos(x.tensor)
    return TypedTorchTensor(
        new_tensor, t.cos(x.type),
        aa=compose_unary("cos", x.aa, dtype_name_of(new_tensor)),
    )


def sqrt(x : TypedTorchTensor) -> TypedTorchTensor:
    new_tensor = torch.sqrt(x.tensor)
    return TypedTorchTensor(
        new_tensor, t.sqrt(x.type),
        aa=compose_unary("sqrt", x.aa, dtype_name_of(new_tensor)),
    )


def maximum(x : TypedTorchTensor, y : TypedTorchTensor) -> TypedTorchTensor:
    return _binary_op_helper(x, y, "max")


def minimum(x : TypedTorchTensor, y : TypedTorchTensor) -> TypedTorchTensor:
    """
    `min(x, y) = -max(-x, -y)`. Matches the `minimum(...)` spec
    keyword's ET lowering so verification across the spec / actual
    boundary normalizes identically.
    """
    return _binary_op_helper(
        0.0, _binary_op_helper(0.0 - x, 0.0 - y, "max"), "-",
    )


def abs(x : TypedTorchTensor) -> TypedTorchTensor:
    """
    `abs(x) = max(x, -x)`. Matches the `abs(...)` spec keyword.
    """
    return _binary_op_helper(x, 0.0 - x, "max")


def relu(x : TypedTorchTensor) -> TypedTorchTensor:
    """
    `relu(x) = max(x, 0)`. Matches the `relu(...)` spec keyword.
    """
    return _binary_op_helper(x, 0.0, "max")


def einsum(x : TypedTorchTensor, y : TypedTorchTensor, einstr : str) -> TypedTorchTensor:
    new_tensor = einops.einsum(x.tensor, y.tensor, einstr)
    new_type = t.einsum(x.type, y.type, einstr)
    new_aa = compose_einsum(
        x.aa, y.aa, x.type, y.type, einstr, active_hardware(),
        x_dtype=dtype_name_of(x.tensor), y_dtype=dtype_name_of(y.tensor),
    )
    return TypedTorchTensor(new_tensor, new_type, aa=new_aa)


class TypedResult:
    def __init__(self, spec : str, device : str = "cpu"):
        self.expected_type = parse_spec_into_type(spec)
        self.shape = tuple(dim_size(d) for d in self.expected_type.st) if self.expected_type.st is not None else tuple()
        self.tensor = torch.zeros(self.shape, device=device)

    def assign(self, result : TypedTorchTensor):
        if not verify_types_equivalent(
                self.expected_type,
                result.type,
        ):
            raise ValueError(f"Attempted to assign a tensor that does not match the spec! Expected : {self.expected_type}, actual : {result.type}")

        slice_expr = []
        for d in result.type.st:
            ds, de = dim_start(d), dim_end(d)
            slice_expr.append(slice(ds, de))
        self.tensor[tuple(slice_expr)] = result.tensor


def zeros(shape : tuple[FullDim, ...], device : str = "cpu") -> TypedTorchTensor:
    torch_shape = tuple(dim_size(d) for d in shape)
    tensor = torch.zeros(torch_shape, device=device)
    type = Type(
        st=shape,
        et=0.0,
    )
    return TypedTorchTensor(tensor, type)


def tensor(
    arr : "torch.Tensor", *shape : FullDim, name : str,
) -> TypedTorchTensor:
    """
    Wrap `arr` as a typed tensor with the given dim shape and a stile
    `Tensor(name=...)` ET. Mirrors `tjax.tensor` for the torch host —
    saves the `TypedTorchTensor(arr, Type(...))` boilerplate.
    """
    return TypedTorchTensor(
        arr, Type(st=shape, et=t.Tensor(dims=shape, name=name)),
    )


def runtime_index(
    name : str,
    dim : FullDim,
    *,
    values_in : FullDim | None = None,
    arr : "torch.Tensor | None" = None,
    permutation : bool = False,
    partition : bool = False,
    device : str = "cuda",
) -> TypedTorchTensor:
    """
    Torch parallel of `tjax.runtime_index`. A 1-d integer tensor used
    as a runtime gather index into another tensor's `values_in` dim.
    Registers algebraic properties (`permutation` / `partition`) with
    the verifier so the surrounding gather/scatter rewrites apply.
    """
    from ..indexing import declare_index_properties
    props = []
    if permutation:
        props.append("permutation")
    if partition:
        props.append("partition")
    if props:
        declare_index_properties(name, *props)
    typ = Type(st=(dim,), et=t.Tensor(dims=(dim,), name=name))
    if arr is None:
        size = dim_size(dim)
        arr = torch.arange(size, dtype=torch.int32, device=device)
    return TypedTorchTensor(arr, typ)


def gather(
    src : TypedTorchTensor, dim : FullDim, idx : TypedTorchTensor,
) -> TypedTorchTensor:
    """Gather `src` along `dim` using `idx` (a 1-D index tensor over
    `dim`). Mirrors `TypedJaxArray.gather`."""
    new_type = src.type.gather(dim, idx.type)
    axis = next(
        i for i, d in enumerate(src.type.st) if dim_name(d) == dim_name(dim)
    )
    idx_i64 = idx.tensor.to(torch.int64)
    # Broadcast idx to src's shape along the gather axis.
    bcast_shape = list(src.tensor.shape)
    expand_shape = [1] * src.tensor.ndim
    expand_shape[axis] = src.tensor.shape[axis]
    idx_bcast = idx_i64.view(expand_shape).expand(bcast_shape)
    new_tensor = torch.gather(src.tensor, axis, idx_bcast)
    return TypedTorchTensor(new_tensor, new_type)
