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
    Type, ShapeType, Tensor, Constant, BinaryOp, UnaryOp, Sliced,
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
    consts : "dict[str, int] | None" = None,
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
        _verify_kernel(
            fn_def, inputs, spec, out_shape, out_dtype, fn,
            consts=consts or {},
        )

        # Re-emit: strip stile decorator + input annotations, add
        # @triton.jit, exec to define the runtime kernel.
        triton_fn = _emit_triton_fn(fn_def, fn, inputs, out_shape, out_dtype)

        return TypedTritonKernel(
            triton_fn, spec, inputs, out_shape, out_dtype,
            consts=consts or {},
        )

    return decorate


class _ProgramIdMarker:
    """Sentinel returned by `_interpret_expr` for `tl.program_id(N)`.
    The surrounding `Assign(target=Name, value=ProgramId)` binds the
    target name as a fresh `SymbolicInt` in env so subsequent
    `target * BLOCK` arithmetic resolves symbolically."""
    pass


def _is_ttl_call(node, name : str) -> bool:
    """True iff `node` is a call of the form `ttl.<name>(…)`."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ttl"
        and node.func.attr == name
    )


def _verify_kernel(
    fn_def, inputs, spec, out_shape, out_dtype, original_fn, consts=None,
):
    """
    Abstract-interpret the kernel's body to build per-variable stile
    Types, then check each store against the spec. Handles:
      - `Assign`: `x = <expr>` binds env[x] = expr's stile Type.
      - `AugAssign`: `acc += <expr>` updates env[acc] via the op.
      - `For` over `range(lo, hi, step)` with concrete bounds: unroll
        and walk the body once per iteration with the loop var bound
        to the concrete int. `_interpret_expr` then resolves slice
        bounds to concrete `Sliced` dims, and the verifier's existing
        tile-merge folds the sum-of-slices into a full reduce.
      - `Expr(Call)` of `tl.store` / `ttl.store` against the output
        pointer: verifies the stored value's Type against the spec
        (tile-restricted to the slice).

    `dim_atoms` maps each Python identifier visible in the kernel's
    closure to its `FullDim` (so `K[lo:hi]` resolves the dim atom by
    Python name, and `range(0, K, BLOCK_K)` resolves `K` and
    `BLOCK_K` to concrete ints for unrolling).
    """
    spec_type = parse_spec_into_type(spec)

    input_types : dict[str, Type] = {}
    for ptr_name, ptr_spec in inputs.items():
        input_types[ptr_name] = parse_spec_into_type(ptr_spec)

    arg_names = [a.arg for a in fn_def.args.args]
    out_ptr_name = arg_names[len(inputs)]

    dim_atoms = _collect_dim_atoms(original_fn)
    env : dict[str, object] = {}
    if consts is not None:
        env.update(consts)

    def visit(stmt):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0].id
            val = _interpret_expr(stmt.value, env, input_types, dim_atoms)
            if isinstance(val, _ProgramIdMarker):
                from ..indexing import SymbolicInt
                env[target] = SymbolicInt(name=target)
            elif val is not None:
                env[target] = val
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            target = stmt.target.id
            rhs = _interpret_expr(stmt.value, env, input_types, dim_atoms)
            if target in env and rhs is not None:
                op_str = {
                    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                }.get(type(stmt.op))
                if op_str is not None:
                    env[target] = t.type_from_binary_op(env[target], rhs, op_str)
        elif isinstance(stmt, ast.For):
            _unroll_for(stmt, env, visit, dim_atoms)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if _is_tl_call(call, "store"):
                store_ptr = call.args[0]
                store_val = call.args[1]
                base = _peel_pointer_base(store_ptr)
                if base == out_ptr_name:
                    _verify_stored_value(
                        store_val, env, input_types, dim_atoms,
                        spec_type, slices=None, out_shape=out_shape,
                    )
            elif _is_ttl_call(call, "store"):
                store_ptr = call.args[0]
                store_val = call.args[1]
                slices = call.args[2:]
                if (
                    isinstance(store_ptr, ast.Name)
                    and store_ptr.id == out_ptr_name
                ):
                    _verify_stored_value(
                        store_val, env, input_types, dim_atoms,
                        spec_type, slices=slices, out_shape=out_shape,
                    )

    for stmt in fn_def.body:
        visit(stmt)


def _collect_dim_atoms(fn) -> dict:
    """Build `name -> FullDim` for every closure / global that's a
    stile dim. Used by the AST-to-SymbolicIndex evaluator to resolve
    things like `K[lo:hi]` and `range(0, K, BLOCK_K)`."""
    out : dict = {}
    closure = fn.__closure__ or ()
    for name, cell in zip(fn.__code__.co_freevars, closure):
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        if hasattr(val, "name") and hasattr(val, "size") and isinstance(val.size, int):
            out[name] = val
    for name, val in fn.__globals__.items():
        if hasattr(val, "name") and hasattr(val, "size") and isinstance(val.size, int):
            out.setdefault(name, val)
    return out


def _unroll_for(node, env, visit, dim_atoms):
    """
    Concretely unroll a `for <var> in range(lo, hi, step):` over
    integer bounds. Within each iteration the loop var is bound in
    `env` as a Python int — so `ttl.load(p, K[ki:ki+BLOCK])` resolves
    to a concrete `Sliced(K, ki, ki+BLOCK)`.
    """
    if not isinstance(node.target, ast.Name):
        return
    if not (
        isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
    ):
        return
    bounds = [_eval_int(a, env, dim_atoms) for a in node.iter.args]
    if any(b is None for b in bounds):
        return
    if len(bounds) == 1:
        lo, hi, step = 0, bounds[0], 1
    elif len(bounds) == 2:
        lo, hi, step = bounds[0], bounds[1], 1
    elif len(bounds) == 3:
        lo, hi, step = bounds
    else:
        return
    var = node.target.id
    saved = env.get(var, None)
    for i in range(lo, hi, step):
        env[var] = i
        for stmt in node.body:
            visit(stmt)
    if saved is None:
        env.pop(var, None)
    else:
        env[var] = saved


def _eval_int(node, env, dim_atoms) -> "int | None":
    """
    Evaluate an AST expression to a Python `int` when possible.
    Handles literals, names (env or dim_atoms[.size]), and the four
    arithmetic ops. Returns `None` if any sub-expression isn't
    concretely an int — caller skips the unrolling.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env and isinstance(env[node.id], int):
            return env[node.id]
        if node.id in dim_atoms:
            return dim_atoms[node.id].size
        return None
    if isinstance(node, ast.BinOp):
        l = _eval_int(node.left, env, dim_atoms)
        r = _eval_int(node.right, env, dim_atoms)
        if l is None or r is None:
            return None
        if isinstance(node.op, ast.Add): return l + r
        if isinstance(node.op, ast.Sub): return l - r
        if isinstance(node.op, ast.Mult): return l * r
        if isinstance(node.op, ast.FloorDiv): return l // r
    return None


def _verify_stored_value(
    value_node, env, input_types, dim_atoms, spec_type, slices, out_shape,
):
    """
    Verify a kernel's output value against the spec, tile-restricting
    the spec to the slice when `slices` is provided. `slices` is a
    list of `DIM[lo:hi]` Subscripts from a `ttl.store` call; `None`
    for a raw `tl.store` (no slice info — verify against the full
    spec).
    """
    val_type = _interpret_expr(value_node, env, input_types, dim_atoms)
    if not isinstance(val_type, Type):
        raise ValueError(
            f"store value `{ast.unparse(value_node)}` does not resolve "
            f"to a typed stile expression"
        )
    target_spec = spec_type
    if slices is not None:
        target_spec = _restrict_spec_to_tile(
            spec_type, slices, out_shape, env, dim_atoms,
        )
    if not verify_exprs_equivalent(val_type.et, target_spec.et):
        raise AssertionError(
            f"Triton kernel output does not match spec.\n"
            f"  spec: {target_spec.et}\n"
            f"  actual: {val_type.et}"
        )


def _restrict_spec_to_tile(spec_type, slices, out_shape, env, dim_atoms):
    """
    Replace each FullDim of `spec_type` whose name matches a slice's
    DIM with the slice's `Sliced(DIM, lo, hi)`. Lets the per-tile
    store certify "this tile equals the spec restricted to (M_slice,
    N_slice, …)".
    """
    overrides = []
    for s in slices:
        if not (isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Slice)):
            continue
        dim_node = s.value
        if not isinstance(dim_node, ast.Name):
            continue
        dim_atom = dim_atoms.get(dim_node.id)
        if dim_atom is None:
            continue
        lo = _eval_symindex(s.slice.lower, env, dim_atoms)
        hi = _eval_symindex(s.slice.upper, env, dim_atoms)
        if lo is None or hi is None:
            continue
        overrides.append(Sliced(dim_atom, lo, hi))
    if not overrides:
        return spec_type
    from ..type import override_dims_in_type
    return override_dims_in_type(spec_type, *overrides)


def _eval_symindex(node, env, dim_atoms):
    """
    Evaluate an AST expression to a stile `SymbolicIndex` — a Python
    `int`, a `SymbolicInt` (for runtime values like `tl.program_id`),
    or an `AffineExpr` for arithmetic over the two. Unknown names
    default to a fresh `SymbolicInt(name)` so e.g. `pid_m * BLOCK_M`
    yields `AffineExpr(0, {(pid_m, BLOCK_M)})`.
    """
    from ..indexing import SymbolicInt, to_affine
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in dim_atoms:
            return dim_atoms[node.id].size
        return SymbolicInt(name=node.id)
    if isinstance(node, ast.BinOp):
        l = _eval_symindex(node.left, env, dim_atoms)
        r = _eval_symindex(node.right, env, dim_atoms)
        if l is None or r is None:
            return None
        if isinstance(node.op, ast.Add): return to_affine(l) + to_affine(r)
        if isinstance(node.op, ast.Sub): return to_affine(l) - to_affine(r)
        if isinstance(node.op, ast.Mult):
            # SymbolicInt only supports multiplication by a Python int —
            # require one side to reduce to an int.
            l_int = _eval_int(node.left, env, dim_atoms)
            r_int = _eval_int(node.right, env, dim_atoms)
            if l_int is not None:
                return to_affine(r) * l_int
            if r_int is not None:
                return to_affine(l) * r_int
            return None
    return None


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


def _interpret_expr(node, env, input_types, dim_atoms):
    """
    Abstract-interpret an AST expression. Returns one of:
      - a stile `Type` (when the expression has a typed-stile result),
      - a raw `float` / `int` (numeric literals or evaluated names —
        passed as-is to `type_from_binary_op` so they get wrapped as
        `Constant(x)` on the spot),
      - `None` (expression isn't stile-relevant).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        # First env (per-loop-iter bindings; loop vars are ints), then
        # dim_atoms (FullDim.size as int).
        if node.id in env:
            v = env[node.id]
            if isinstance(v, int):
                return float(v)
            if isinstance(v, Type):
                return v
            # SymbolicInt / AffineExpr — not a Type, doesn't flow as
            # a value expression in this slot.
            return None
        if node.id in dim_atoms:
            return float(dim_atoms[node.id].size)
        return None
    if _is_tl_call(node, "program_id"):
        # `pid_x = tl.program_id(N)` — a runtime grid coord. Treat as
        # an opaque `SymbolicInt`; the variable name (from the
        # surrounding `Assign`) is used as the atom's identity. Caller
        # handles the binding in env.
        return _ProgramIdMarker()
    if _is_tl_call(node, "load"):
        ptr_base = _peel_pointer_base(node.args[0])
        if ptr_base is not None and ptr_base in input_types:
            return input_types[ptr_base]
        return None
    if _is_ttl_call(node, "load"):
        # `ttl.load(ptr, DIM_0[lo:hi], DIM_1[lo:hi], …)`. Restrict the
        # input's Type to the slice: each `DIM[lo:hi]` becomes
        # `Sliced(DIM, lo_int, hi_int)`, evaluated from the AST against
        # the current env / dim_atoms.
        if not (len(node.args) >= 1 and isinstance(node.args[0], ast.Name)):
            return None
        ptr_name = node.args[0].id
        if ptr_name not in input_types:
            return None
        base = input_types[ptr_name]
        slice_args = node.args[1:]
        if len(slice_args) != len(base.st):
            return base
        overrides = []
        for s in slice_args:
            if not (isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Slice)):
                return base
            dim_node = s.value
            if not isinstance(dim_node, ast.Name):
                return base
            dim_atom = dim_atoms.get(dim_node.id)
            if dim_atom is None:
                return base
            lo = _eval_symindex(s.slice.lower, env, dim_atoms)
            hi = _eval_symindex(s.slice.upper, env, dim_atoms)
            if lo is None or hi is None:
                return base
            overrides.append(Sliced(dim_atom, lo, hi))
        from ..type import override_dims_in_type
        return override_dims_in_type(base, *overrides)
    if _is_tl_call(node, "dot"):
        a_type = _interpret_expr(node.args[0], env, input_types, dim_atoms)
        b_type = _interpret_expr(node.args[1], env, input_types, dim_atoms)
        if not (isinstance(a_type, Type) and isinstance(b_type, Type)):
            return None
        if len(a_type.st) != 2 or len(b_type.st) != 2:
            return None
        m_dim, k_dim = a_type.st
        kb_dim, n_dim = b_type.st
        if dim_name(k_dim) != dim_name(kb_dim):
            return None
        einstr = (
            f"{dim_name(m_dim)} {dim_name(k_dim)}, "
            f"{dim_name(kb_dim)} {dim_name(n_dim)} -> "
            f"{dim_name(m_dim)} {dim_name(n_dim)}"
        )
        return t.einsum(a_type, b_type, einstr)
    if _is_tl_call(node, "zeros") or _is_ttl_call(node, "zeros"):
        # `ttl.zeros(DIM_0[lo:hi], DIM_1[lo:hi], …, dtype=…)` returns
        # the additive identity at the given (sliced) shape — used as
        # an accumulator init. `tl.zeros((BLOCK_M, BLOCK_N), …)` is
        # also accepted but doesn't carry stile-dim info, so the
        # caller gets a shape-agnostic Constant(0).
        if _is_ttl_call(node, "zeros"):
            overrides = []
            for s in node.args:
                if not (isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Slice)):
                    return None
                dim_node = s.value
                if not isinstance(dim_node, ast.Name):
                    return None
                dim_atom = dim_atoms.get(dim_node.id)
                if dim_atom is None:
                    return None
                lo = _eval_symindex(s.slice.lower, env, dim_atoms)
                hi = _eval_symindex(s.slice.upper, env, dim_atoms)
                if lo is None or hi is None:
                    return None
                overrides.append(Sliced(dim_atom, lo, hi))
            return Type(st=tuple(overrides), et=Constant(0.0))
        # Plain `tl.zeros((...), dtype=…)`: shape-agnostic zero.
        return Type(st=(), et=Constant(0.0))
    if _is_tl_call(node, "exp"):
        child_type = _interpret_expr(node.args[0], env, input_types, dim_atoms)
        if not isinstance(child_type, Type):
            return None
        return t.exp(child_type)
    if isinstance(node, ast.BinOp):
        l = _interpret_expr(node.left, env, input_types, dim_atoms)
        r = _interpret_expr(node.right, env, input_types, dim_atoms)
        if l is None or r is None:
            return None
        op_str = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
        }.get(type(node.op))
        if op_str is None:
            return None
        return t.type_from_binary_op(l, r, op_str)
    return None


class _RewriteTtlCalls(ast.NodeTransformer):
    """
    Source-level rewriter for the stile-flavored slice ops:

      ttl.load(ptr, DIM_0[lo_0:hi_0], DIM_1[lo_1:hi_1], …)
      ttl.store(ptr, value, DIM_0[lo_0:hi_0], DIM_1[lo_1:hi_1], …)

    Each `DIM_i[lo_i:hi_i]` slices the ptr along its `DIM_i` axis. The
    rewriter emits an offsets array per axis, broadcasts them into a
    multi-dim pointer offset using row-major strides derived from the
    input's declared shape, and wraps in `tl.load` / `tl.store`. The
    per-axis size is extracted syntactically — Triton's `tl.arange`
    needs a constexpr upper bound, so `hi - lo` must reduce to a name
    or literal via one of these patterns:
      - `0 : N`         → N
      - `start : start + N` → N
      - `k*N : (k+1)*N` → N
    """
    def __init__(self, input_types : dict):
        self.input_types = input_types
        self._uid = 0

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            if _is_ttl_call(node.value, "load"):
                return self._rewrite_load_assign(node)
            if _is_ttl_call(node.value, "zeros"):
                return self._rewrite_zeros_assign(node)
        return node

    def _rewrite_zeros_assign(self, node):
        """
        `acc = ttl.zeros(M[lo:hi], N[lo:hi], dtype=…)`
        →
        `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)`

        Sizes are pulled from the same `_extract_size` patterns the
        load/store rewriter uses. dtype defaults to `tl.float32`.
        """
        call = node.value
        sizes = []
        for s in call.args:
            if not (isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Slice)):
                return node
            size = _extract_size(s.slice.lower, s.slice.upper)
            if size is None:
                return node
            sizes.append(size)
        shape = ast.Tuple(elts=sizes, ctx=ast.Load())
        dtype_kw = next(
            (kw for kw in call.keywords if kw.arg == "dtype"),
            ast.keyword(
                arg="dtype",
                value=ast.Attribute(
                    value=ast.Name(id="tl"), attr="float32",
                ),
            ),
        )
        new_call = ast.Call(
            func=ast.Attribute(value=ast.Name(id="tl"), attr="zeros"),
            args=[shape],
            keywords=[dtype_kw],
        )
        return ast.Assign(targets=node.targets, value=new_call)

    def visit_Expr(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Call)
            and _is_ttl_call(node.value, "store")
        ):
            return self._rewrite_store_stmt(node)
        return node

    def _next_uid(self):
        u = self._uid
        self._uid += 1
        return u

    def _rewrite_load_assign(self, node):
        call = node.value
        target = node.targets[0]
        ptr = call.args[0]
        slices = call.args[1:]
        prelude, ptr_offs, mask = self._lower_slices(ptr, slices)
        load_call = ast.Call(
            func=ast.Attribute(value=ast.Name(id="tl"), attr="load"),
            args=[ast.BinOp(left=ptr, op=ast.Add(), right=ptr_offs)],
            keywords=[ast.keyword(arg="mask", value=mask)],
        )
        return prelude + [ast.Assign(targets=[target], value=load_call)]

    def _rewrite_store_stmt(self, node):
        call = node.value
        ptr = call.args[0]
        value = call.args[1]
        slices = call.args[2:]
        prelude, ptr_offs, mask = self._lower_slices(ptr, slices)
        store_call = ast.Call(
            func=ast.Attribute(value=ast.Name(id="tl"), attr="store"),
            args=[
                ast.BinOp(left=ptr, op=ast.Add(), right=ptr_offs),
                value,
            ],
            keywords=[ast.keyword(arg="mask", value=mask)],
        )
        return prelude + [ast.Expr(value=store_call)]

    def _lower_slices(self, ptr, slices):
        """
        Build the prelude `_offs_<uid>_<i>` assignments and the
        combined `ptr_offs` / `mask` expressions for an N-dim
        ttl.load/store. Strides come from the declared input shape
        (row-major; outer dim has the largest stride).
        """
        if not isinstance(ptr, ast.Name):
            raise ValueError(
                f"ttl.load/store ptr must be a plain name (got "
                f"`{ast.unparse(ptr)}`)"
            )
        input_type = self.input_types.get(ptr.id)
        if input_type is None:
            raise ValueError(
                f"ttl.load/store: ptr `{ptr.id}` not declared in `inputs=`"
            )
        if len(slices) != len(input_type.st):
            raise ValueError(
                f"ttl.load/store: ptr `{ptr.id}` has {len(input_type.st)} "
                f"dims but {len(slices)} slices were given"
            )

        uid = self._next_uid()
        n = len(slices)
        # Parse each slice: extract (dim_name, lo, hi).
        dims_los_his = []
        for i, s in enumerate(slices):
            if not (isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Slice)):
                raise ValueError(
                    f"ttl.load/store slice arg {i} must be `DIM[lo:hi]` "
                    f"(got `{ast.unparse(s)}`)"
                )
            dim_node = s.value
            if not isinstance(dim_node, ast.Name):
                raise ValueError(
                    f"ttl.load/store slice arg {i}: dim must be a name"
                )
            lo, hi = s.slice.lower, s.slice.upper
            if lo is None or hi is None:
                raise ValueError(
                    f"ttl.load/store slice arg {i}: must have both "
                    f"lower and upper bound"
                )
            dims_los_his.append((dim_node, lo, hi))

        # Prelude: one per-axis offsets var.
        prelude = []
        offs_names = []
        for i, (dim_node, lo, hi) in enumerate(dims_los_his):
            size = _extract_size(lo, hi)
            if size is None:
                raise ValueError(
                    f"ttl.load/store slice arg {i}: can't extract block "
                    f"size from `{ast.unparse(lo)} : {ast.unparse(hi)}`. "
                    f"Use one of: `0:N`, `start:start+N`, or "
                    f"`k*N:(k+1)*N`."
                )
            arange = ast.Call(
                func=ast.Attribute(value=ast.Name(id="tl"), attr="arange"),
                args=[ast.Constant(value=0), size],
                keywords=[],
            )
            offs_expr = ast.BinOp(left=lo, op=ast.Add(), right=arange)
            offs_name = f"_ttl_offs_{uid}_{i}"
            prelude.append(ast.Assign(
                targets=[ast.Name(id=offs_name, ctx=ast.Store())],
                value=offs_expr,
            ))
            offs_names.append(offs_name)

        all_dim_nodes = [d for d, _, _ in dims_los_his]

        # Combined pointer offset: Σ offs_i_bcast * stride_i.
        # Row-major: stride_i = product of dim sizes for axes after i.
        def offs_bcast(i):
            if n == 1:
                return ast.Name(id=offs_names[i], ctx=ast.Load())
            slc_elts = [
                ast.Slice() if j == i else ast.Constant(value=None)
                for j in range(n)
            ]
            return ast.Subscript(
                value=ast.Name(id=offs_names[i], ctx=ast.Load()),
                slice=ast.Tuple(elts=slc_elts, ctx=ast.Load()),
                ctx=ast.Load(),
            )

        terms = []
        for i in range(n):
            ob = offs_bcast(i)
            stride = _stride_for_axis(i, all_dim_nodes)
            terms.append(
                ast.BinOp(left=ob, op=ast.Mult(), right=stride)
                if stride is not None else ob
            )
        ptr_offs = terms[0]
        for t in terms[1:]:
            ptr_offs = ast.BinOp(left=ptr_offs, op=ast.Add(), right=t)

        # Mask: AND of (offs_i_bcast < DIM_i).
        mask_terms = [
            ast.Compare(
                left=offs_bcast(i), ops=[ast.Lt()],
                comparators=[all_dim_nodes[i]],
            )
            for i in range(n)
        ]
        mask = mask_terms[0]
        for m in mask_terms[1:]:
            mask = ast.BinOp(left=mask, op=ast.BitAnd(), right=m)

        return prelude, ptr_offs, mask


def _extract_size(lo, hi):
    """
    Try to syntactically reduce `hi - lo` to a constexpr-friendly AST
    node Triton's `tl.arange` will accept. Supports:
      - `0 : N`              → N
      - `start : start + N`  → N
      - `k*N : (k+1)*N`      → N
    Returns the size AST or `None` if no pattern matched.
    """
    # 0 : N
    if isinstance(lo, ast.Constant) and lo.value == 0:
        return hi
    # start : start + N
    if isinstance(hi, ast.BinOp) and isinstance(hi.op, ast.Add):
        if ast.dump(hi.left) == ast.dump(lo):
            return hi.right
        if ast.dump(hi.right) == ast.dump(lo):
            return hi.left
    # k*N : (k+1)*N
    if (
        isinstance(lo, ast.BinOp) and isinstance(lo.op, ast.Mult)
        and isinstance(hi, ast.BinOp) and isinstance(hi.op, ast.Mult)
        and ast.dump(hi.right) == ast.dump(lo.right)
        and isinstance(hi.left, ast.BinOp) and isinstance(hi.left.op, ast.Add)
        and ast.dump(hi.left.left) == ast.dump(lo.left)
        and isinstance(hi.left.right, ast.Constant)
        and hi.left.right.value == 1
    ):
        return lo.right
    return None


def _stride_for_axis(axis : int, dim_nodes : list) -> "ast.AST | None":
    """
    Row-major stride for `axis`: product of dim sizes for axes
    after `axis`. The innermost (last) axis has stride 1, signaled
    by returning `None` so the caller skips the multiplication.
    """
    later = dim_nodes[axis + 1:]
    if not later:
        return None
    expr = later[0]
    for d in later[1:]:
        expr = ast.BinOp(left=expr, op=ast.Mult(), right=d)
    return expr


def _emit_triton_fn(fn_def, original_fn, inputs, out_shape=None, out_dtype=None):
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
    # Rewrite `ttl.load`/`ttl.store` into raw `tl.load`/`tl.store` with
    # offsets + masks computed from the slice info + row-major strides
    # derived from the declared input shape. The output pointer's
    # shape comes from the @ttl.jit decorator's out_shape; the output
    # is always the kernel-arg right after the inputs.
    ptr_types : dict[str, Type] = {
        ptr_name: parse_spec_into_type(ptr_spec)
        for ptr_name, ptr_spec in inputs.items()
    }
    arg_names = [a.arg for a in fn_def.args.args]
    out_ptr_name = arg_names[len(inputs)]
    out_full_type = parse_spec_into_type(
        " ".join(d.name for d in out_shape)
    )
    out_full_type = Type(out_shape, out_full_type.et, out_dtype)
    ptr_types[out_ptr_name] = out_full_type
    fn_def = _RewriteTtlCalls(ptr_types).visit(fn_def)
    ast.fix_missing_locations(fn_def)
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
    `consts` declared at decoration time are passed as kwargs to
    `@triton.jit` automatically — overridable per-launch.
    """
    triton_fn : object
    spec : str
    inputs : dict[str, str]
    out_shape : ShapeType
    out_dtype : object
    consts : dict

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
        all_kwargs = dict(self.kernel.consts)
        all_kwargs.update(kwargs)
        self.kernel.triton_fn[self.grid](*raw_inputs, out_tensor, **all_kwargs)
        spec_type = parse_spec_into_type(self.kernel.spec)
        return TypedTorchTensor(
            out_tensor,
            Type(self.kernel.out_shape, spec_type.et, self.kernel.out_dtype),
        )
