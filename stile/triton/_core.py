try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

import ast
import importlib.util
import inspect
import os
import tempfile
import textwrap
from dataclasses import dataclass

import stile.type as t
from ..type import (
    Type, ShapeType, Tensor, Constant, BinaryOp, UnaryOp,
    dim_size, dim_full_dim, dim_name, as_int,
)
from ..specification import parse_spec_into_type
from ..verification import verify_exprs_equivalent
from ..torch._core import TypedTorchTensor


def jit(
    *,
    spec : str,
    inputs : dict[str, str],
    out_shape : ShapeType,
    out_dtype : object = None,
):
    """
    Decorator producing a typed Triton kernel.

    The user writes a kernel that *looks like Triton* — `tl.load` /
    `tl.store` / arithmetic / `tl.exp` / etc. — and declares per-pointer
    stile types via `inputs={...}` (mapping each pointer parameter name
    to a stile spec like `"X:N"`) plus `spec=` for what the output
    should equal.

    At decoration time the decorator:
      1. `inspect.getsource(fn)` + `ast.parse` the body.
      2. Abstract-interpret the AST: every `tl.load(ptr + …)` binds to
         the input's stile `Tensor(name=…)`; arithmetic / `tl.exp` /
         etc. propagate stile Types. At each `tl.store`, verifies the
         stored value's stile ET against the OutputSpec.
      3. Re-emits the function with stile decorator + `inputs=`
         stripped, decorates with `@triton.jit`, and `exec`s it. The
         returned object is `TypedTritonKernel`, launchable with
         Triton's familiar `kernel[grid](*typed_inputs)` syntax.
    """
    def decorate(fn):
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        fn_def = tree.body[0]
        assert isinstance(fn_def, ast.FunctionDef), (
            "@ttl.jit must decorate a plain function definition"
        )

        # Verification: abstract-interpret the body. Raises if any
        # `tl.store` writes a value whose stile ET doesn't match the
        # declared spec.
        _verify_kernel(fn_def, inputs, spec, out_shape, out_dtype)

        # Re-emit: strip stile decorator + input annotations, add
        # @triton.jit, exec to define the runtime kernel.
        triton_fn = _emit_triton_fn(fn_def, fn)

        return TypedTritonKernel(
            triton_fn, spec, inputs, out_shape, out_dtype,
        )

    return decorate


def _verify_kernel(fn_def, inputs, spec, out_shape, out_dtype):
    """
    Abstract-interpret the kernel's body to build per-variable stile
    Types, then check each `tl.store(...)`'s value against the spec.

    Pattern recognition (first pass, kernel-by-kernel):
      - Assignment `x = tl.load(p + offs, mask=…)` → if `p` is one of
        the declared input pointers, bind `x` to that input's stile
        `Tensor` ET.
      - Assignment `x = <arith on tracked vars or constants>` → build
        a `BinaryOp` / `UnaryOp` / `Constant` ET.
      - `tl.store(p_out + offs, value, mask=…)` → if `p_out` is the
        output pointer, verify `value`'s stile ET against the spec.
    """
    spec_type = parse_spec_into_type(spec)

    # Resolve `inputs={"X_ptr": "X:N"}` into a map from pointer-arg
    # name → stile `Type`.
    input_types : dict[str, Type] = {}
    for ptr_name, ptr_spec in inputs.items():
        input_types[ptr_name] = parse_spec_into_type(ptr_spec)

    # The output pointer is whatever positional arg comes immediately
    # after the input pointers. Convention: first N args are pointers
    # (N = len(inputs)), next is the output pointer, then any constexpr
    # / scalar args.
    arg_names = [a.arg for a in fn_def.args.args]
    out_ptr_name = arg_names[len(inputs)]

    env : dict[str, Type] = {}

    for stmt in fn_def.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0].id
            et = _interpret_expr(stmt.value, env, input_types)
            if et is not None:
                env[target] = et
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            # tl.store(p_out + offs, value, mask=…) — verification point.
            if _is_tl_call(call, "store"):
                store_ptr = call.args[0]
                store_val = call.args[1]
                # The pointer expression is `out_ptr + offs` — peel the +
                # to identify the base pointer.
                base = _peel_pointer_base(store_ptr)
                if base == out_ptr_name:
                    val_type = _interpret_expr(store_val, env, input_types)
                    if val_type is None:
                        raise ValueError(
                            f"tl.store value `{ast.unparse(store_val)}` "
                            f"does not resolve to a typed stile expression"
                        )
                    if not verify_exprs_equivalent(val_type.et, spec_type.et):
                        raise AssertionError(
                            f"Triton kernel output does not match spec.\n"
                            f"  spec: {spec_type.et}\n"
                            f"  actual: {val_type.et}"
                        )


def _is_tl_call(node, name : str) -> bool:
    """True iff `node` is a call of the form `tl.<name>(…)`."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == name
    )


def _peel_pointer_base(node) -> "str | None":
    """`out_ptr + offs` → `"out_ptr"`. Recurse through `+` /
    parenthesizations so we still find the base under a nested
    expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _peel_pointer_base(node.left)
        if left is not None:
            return left
        return _peel_pointer_base(node.right)
    return None


def _interpret_expr(node, env, input_types):
    """
    Abstract-interpret an AST expression. Returns one of:
      - a stile `Type` (when the expression has a typed-stile result),
      - a raw `float` / `int` (numeric literals — passed as-is to
        `type_from_binary_op` so they get wrapped as `Constant(x)` on
        the spot),
      - `None` (expression isn't stile-relevant, e.g. plain ints used
        as block sizes / offsets).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if _is_tl_call(node, "load"):
        # `tl.load(ptr + offs, mask=…)` — bind to the declared input
        # type for `ptr`.
        ptr_base = _peel_pointer_base(node.args[0])
        if ptr_base is not None and ptr_base in input_types:
            return input_types[ptr_base]
        return None
    if _is_tl_call(node, "exp"):
        child_type = _interpret_expr(node.args[0], env, input_types)
        if not isinstance(child_type, Type):
            return None
        return t.exp(child_type)
    if isinstance(node, ast.BinOp):
        l = _interpret_expr(node.left, env, input_types)
        r = _interpret_expr(node.right, env, input_types)
        if l is None or r is None:
            return None
        op_str = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
        }.get(type(node.op))
        if op_str is None:
            return None
        return t.type_from_binary_op(l, r, op_str)
    return None


def _emit_triton_fn(fn_def, original_fn):
    """
    Re-emit the kernel function with stile decorations stripped and
    `@triton.jit` applied. Triton's parser uses `inspect.getsourcelines`
    so the function must live in a real `.py` file; we write the
    stripped source to a per-kernel temp file and import it as a
    module. The module gets the original function's globals merged
    in so any free names inside the body (dim sizes, etc.) resolve.
    """
    fn_def.decorator_list = [
        d for d in fn_def.decorator_list
        if not _is_ttl_jit_decorator(d)
    ]
    new_src = ast.unparse(ast.Module(body=[fn_def], type_ignores=[]))

    src_with_imports = (
        "import triton\n"
        "import triton.language as tl\n"
        + new_src
        + "\n"
    )

    tmpdir = tempfile.mkdtemp(prefix="stile_triton_")
    src_path = os.path.join(tmpdir, f"{fn_def.name}.py")
    with open(src_path, "w") as f:
        f.write(src_with_imports)

    spec = importlib.util.spec_from_file_location(fn_def.name, src_path)
    mod = importlib.util.module_from_spec(spec)
    # Merge in user globals so dim sizes etc. resolve, but skip the
    # original module's loader/spec metadata — those would confuse
    # importlib's own bookkeeping for the temp module.
    for k, v in original_fn.__globals__.items():
        if k.startswith("__") and k.endswith("__"):
            continue
        mod.__dict__.setdefault(k, v)
    # Inline closure free vars (e.g. `N = dim("TTN", 128)` declared in
    # the caller's scope) so Triton's parser can resolve them. Stile
    # `FullDim`s lower to their `.size` int — Triton's body sees a
    # plain compile-time constant rather than an opaque Python object.
    closure = original_fn.__closure__ or ()
    for name, cell in zip(original_fn.__code__.co_freevars, closure):
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        if hasattr(val, "size") and hasattr(val, "name") and isinstance(val.size, int):
            val = val.size
        # Triton requires kernel-accessible globals to be wrapped as
        # `tl.constexpr(...)`. Ints / floats lift cleanly; anything
        # else gets dropped through to its raw value.
        if HAS_TRITON and isinstance(val, (int, float)):
            val = tl.constexpr(val)
        mod.__dict__[name] = val
    spec.loader.exec_module(mod)
    raw_fn = getattr(mod, fn_def.name)

    if HAS_TRITON:
        return triton.jit(raw_fn)
    return raw_fn


def _is_ttl_jit_decorator(node) -> bool:
    """Recognize `@ttl.jit(...)` so we know which decorator to strip."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "jit":
            base = func.value
            return isinstance(base, ast.Name) and base.id in ("ttl", "stile_triton")
    return False


@dataclass
class TypedTritonKernel:
    """
    Verified Triton kernel. Triton's `[grid](*args)` launch syntax
    delegates to the underlying `@triton.jit` function; the
    `TypedTorchTensor` inputs are unwrapped to raw tensors before
    launch, and the output is rewrapped with the declared type.
    """
    triton_fn : object
    spec : str
    inputs : dict[str, str]
    out_shape : ShapeType
    out_dtype : object

    def __getitem__(self, grid):
        if isinstance(grid, int):
            grid = (grid,)
        return _Launcher(self, grid)


class _Launcher:
    def __init__(self, kernel : TypedTritonKernel, grid):
        self.kernel = kernel
        self.grid = grid

    def __call__(self, *typed_inputs : "TypedTorchTensor", **kwargs):
        if not HAS_TRITON:
            raise RuntimeError(
                "Triton is not installed; cannot launch the verified "
                "kernel. Install triton or run on a GPU host."
            )
        out_shape_ints = tuple(
            as_int(dim_size(d)) for d in self.kernel.out_shape
        )
        out_dtype = self.kernel.out_dtype or torch.float32
        out_tensor = torch.empty(
            out_shape_ints, dtype=out_dtype, device="cuda",
        )
        raw_inputs = tuple(ti.tensor for ti in typed_inputs)
        self.kernel.triton_fn[self.grid](*raw_inputs, out_tensor, **kwargs)
        spec_type = parse_spec_into_type(self.kernel.spec)
        return TypedTorchTensor(
            out_tensor,
            Type(self.kernel.out_shape, spec_type.et, self.kernel.out_dtype),
        )
