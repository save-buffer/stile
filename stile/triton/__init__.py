from ._core import (
    jit,
    TypedTritonKernel,
    HAS_TRITON,
)


# Kernel intrinsics. These names are only meaningful *inside* an
# `@ttl.jit`-decorated kernel, where the decorator rewrites them from the
# function's source (`ttl.load`/`ttl.store` → typed `tl.load`/`tl.store`
# with computed offsets, `ttl.range` → the verified loop form, etc.). They
# are never executed as plain Python. Defining them as stubs gives the DSL
# surface a real, importable identity — type-checkers and IDEs resolve
# `ttl.load(...)`, and a stray call *outside* a kernel raises a clear error
# instead of an opaque `AttributeError`.
def _kernel_intrinsic(name : str):
    def _stub(*args, **kwargs):
        raise RuntimeError(
            f"`ttl.{name}(...)` is a typed-Triton kernel intrinsic — it only "
            f"has meaning inside an `@ttl.jit`-decorated kernel (the decorator "
            f"rewrites it from source). It can't be called directly."
        )
    _stub.__name__ = name
    _stub.__qualname__ = name
    return _stub


load = _kernel_intrinsic("load")
store = _kernel_intrinsic("store")
range = _kernel_intrinsic("range")
static_range = _kernel_intrinsic("static_range")
zeros = _kernel_intrinsic("zeros")
full = _kernel_intrinsic("full")
mask = _kernel_intrinsic("mask")
gather = _kernel_intrinsic("gather")
