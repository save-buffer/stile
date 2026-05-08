"""
Run typed Pallas kernels on a real CUDA GPU through stile.

What you'll see:
  1. A scalar-multiply kernel and a vector-add kernel verified by
     stile against one-line specs and then lowered to Pallas
     Mosaic-GPU and executed on the GPU.
  2. A negative case: the same scalar-multiply kernel with the wrong
     factor — the verifier rejects it before Pallas ever lowers, so
     no GPU work is launched.

Today's gaps (upstream Pallas):
  - Mosaic-GPU `Lane` lowering doesn't implement `dot_general`, so
    matmul / flash-attention kernels stay on the interpret path
    until that ships. The same typed kernels (see
    `tests/test_typed_pallas.py`) run end-to-end on CPU interpret;
    when upstream lands the missing primitives, no stile changes
    are needed to flip them to GPU.

Run from the project root on a CUDA-equipped host:
  uv run python examples/demo_pallas_gpu.py
"""
import jax
import jax.numpy as jnp

import stile.jax as tjax
import stile.jax.pallas as tpl
from stile import dim, reset_stile


def main():
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    if not any(d.platform == "gpu" for d in devices):
        print(
            "No CUDA device available — this demo needs a GPU. "
            "On a CPU-only host, see `tests/test_typed_pallas.py` for "
            "the same kernels running under Pallas's CPU interpreter."
        )
        return

    # CompilerParams selects the Mosaic-GPU backend (the Pallas-on-GPU
    # path that ships with current jaxlib).
    from jax.experimental.pallas import mosaic_gpu as plgpu
    cp = plgpu.CompilerParams()

    # --- 1. Scalar multiply on GPU --------------------------------------
    print("\n[1] scalar multiply: y = 2*x")
    N = dim("N", 128)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def scalar_mul(x_ref, o_ref):
        o_ref.assign(x_ref.load() * 2)

    y = tpl.typed_pallas_call(
        scalar_mul,
        out_type=tpl.OutputSpec("2 * N", x.type.st, jnp.float32),
        interpret=False,
        compiler_params=cp,
    )(x)
    print(f"    spec    : 2 * N")
    print(f"    device  : {y.arr.device}")
    print(f"    matches : {bool(jnp.allclose(y.arr, x.arr * 2))}")

    reset_stile()

    # --- 2. Vector add on GPU -------------------------------------------
    print("\n[2] vector add: c = a + b   (two distinct labeled inputs)")
    M = dim("M", 128)
    a = tjax.random.normal(jax.random.PRNGKey(0), M, name="a")
    b = tjax.random.normal(jax.random.PRNGKey(1), M, name="b")

    def vector_add(a_ref, b_ref, o_ref):
        o_ref.assign(a_ref.load() + b_ref.load())

    c = tpl.typed_pallas_call(
        vector_add,
        out_type=tpl.OutputSpec("a:M + b:M", a.type.st, jnp.float32),
        interpret=False,
        compiler_params=cp,
    )(a, b)
    print(f"    spec    : a:M + b:M")
    print(f"    device  : {c.arr.device}")
    print(f"    matches : {bool(jnp.allclose(c.arr, a.arr + b.arr))}")

    reset_stile()

    # --- 3. Verifier rejects a wrong kernel BEFORE GPU dispatch ---------
    print("\n[3] negative case: kernel multiplies by 3, spec says 2 — verifier rejects")
    N = dim("N", 128)
    x = tjax.random.normal(jax.random.PRNGKey(0), N)

    def buggy_scalar_mul(x_ref, o_ref):
        o_ref.assign(x_ref.load() * 3)  # wrong factor

    runner = tpl.typed_pallas_call(
        buggy_scalar_mul,
        out_type=tpl.OutputSpec("2 * N", x.type.st, jnp.float32),
        interpret=False,
        compiler_params=cp,
    )
    try:
        runner(x)
        print("    UNEXPECTED: kernel was accepted!")
    except ValueError as e:
        msg = str(e).splitlines()[0]
        print(f"    rejected as expected: {msg}")
        print(f"    (no GPU dispatch happened — verification ran at trace time)")

    print("\nDone.")


if __name__ == "__main__":
    main()
