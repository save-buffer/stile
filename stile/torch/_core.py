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

import einops


class TypedTorchTensor:
    def __init__(self, tensor : torch.Tensor, type : Type):
        self.tensor = tensor
        self.type = type

    @property
    def data_ptr(self) -> int:
        return self.tensor.data_ptr()

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

    return TypedTorchTensor(new_tensor, new_type)


def exp(x : TypedTorchTensor) -> TypedTorchTensor:
    new_type = t.exp(x.type)
    new_tensor = torch.exp(x.tensor)
    return TypedTorchTensor(new_tensor, new_type)


def sin(x : TypedTorchTensor) -> TypedTorchTensor:
    new_type = t.sin(x.type)
    new_tensor = torch.sin(x.tensor)
    return TypedTorchTensor(new_tensor, new_type)


def cos(x : TypedTorchTensor) -> TypedTorchTensor:
    new_type = t.cos(x.type)
    new_tensor = torch.cos(x.tensor)
    return TypedTorchTensor(new_tensor, new_type)


def sqrt(x : TypedTorchTensor) -> TypedTorchTensor:
    new_type = t.sqrt(x.type)
    new_tensor = torch.sqrt(x.tensor)
    return TypedTorchTensor(new_tensor, new_type)


def maximum(x : TypedTorchTensor, y : TypedTorchTensor) -> TypedTorchTensor:
    return _binary_op_helper(x, y, "max")


def einsum(x : TypedTorchTensor, y : TypedTorchTensor, einstr : str) -> TypedTorchTensor:
    new_tensor = einops.einsum(x.tensor, y.tensor, einstr)
    new_type = t.einsum(x.type, y.type, einstr)
    return TypedTorchTensor(new_tensor, new_type)


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
