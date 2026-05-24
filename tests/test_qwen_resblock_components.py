"""
Building blocks of a transformer resblock: RMSNorm, SwiGLU MLP, etc.
Each test asserts both structural equivalence to a stile spec and
numerical equivalence to a jax reference, so any future verifier
change that breaks composition gets caught.
"""
import jax
import jax.numpy as jnp
import pytest

import stile.jax as tjax
from stile import dim, reset_stile


def test_rms_norm(reset):
    """
    RMSNorm: `y_i = x_i * scale_i / sqrt(mean_j(x_j^2) + eps)` along
    the hidden dim. All primitives already exist (`sum`, `sqrt`, `*`,
    `/`), this just confirms they compose to the textbook form.
    """
    n_tokens = dim("n_tokens", 8)
    d_model = dim("d_model", 16)
    eps = 0.0001

    k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
    X = tjax.random.normal(k1, n_tokens, d_model, name="X")
    scale = tjax.random.normal(k2, d_model, name="scale")

    @tjax.jit(
        spec=f"(X:n_tokens d_model * scale:d_model "
             f"/ sqrt(sum[d_model](X:n_tokens d_model * X:n_tokens d_model) "
             f"/ {d_model.size} + {eps})) -> n_tokens d_model"
    )
    def rms_norm(X, scale):
        sq_sum = (X * X).sum(d_model)
        inv_rms = 1.0 / tjax.sqrt(sq_sum / d_model.size + eps)
        broadcast = inv_rms.repeat(d_model).rearrange(n_tokens, d_model)
        scale_full = scale.repeat(n_tokens).rearrange(n_tokens, d_model)
        return X * scale_full * broadcast

    result = rms_norm(X=X, scale=scale)

    # Numerical reference.
    sq_mean = (X.arr * X.arr).mean(axis=-1, keepdims=True)
    expected = X.arr * scale.arr / jnp.sqrt(sq_mean + eps)
    assert jnp.allclose(result.arr, expected, atol=1e-5)


def test_rope(reset):
    """
    Rotary position embedding. The "rotate-half" trick is just a
    permutation + sign-flip on the `dhead` axis:
        R(q)[j] = sign[j] · q[σ(j)]
    with σ = [d/2, …, d−1, 0, …, d/2−1] (cyclic shift by d/2) and
    sign = [−1, …, −1, +1, …, +1] (negate first half).
    Then `rope(q, θ) = q · cos(θ) + R(q) · sin(θ)`. Both the
    permutation and sign array are precomputed constants — the
    kernel reads them via opaque named tensors, so the spec
    references the same tensors by name and the verifier matches
    structurally. Numerically compared to a reference
    `concat([-x_hi, x_lo], axis=-1) * sin + x * cos`.
    """
    n_tokens = dim("n_tokens", 8)
    dhead = dim("dhead", 8)
    dhead_half = dhead.size // 2

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    X = tjax.random.normal(k1, n_tokens, dhead, name="X")
    cos_table = tjax.random.normal(k2, n_tokens, dhead, name="cos_table")
    sin_table = tjax.random.normal(k3, n_tokens, dhead, name="sin_table")

    rope_perm = tjax.runtime_index(
        "rope_perm", dhead, values_in=dhead, permutation=True,
        arr=jnp.concatenate([
            jnp.arange(dhead_half, dhead.size),
            jnp.arange(0, dhead_half),
        ]),
    )
    sign_mask = tjax.tensor(
        jnp.concatenate([
            -jnp.ones(dhead_half), jnp.ones(dhead_half),
        ]).astype(jnp.float32),
        dhead,
        name="sign_mask",
    )

    @tjax.jit(
        spec="(X:n_tokens dhead * cos_table:n_tokens dhead + "
             "gather[dhead](X:n_tokens dhead, rope_perm:dhead) "
             "* sign_mask:dhead * sin_table:n_tokens dhead) -> n_tokens dhead"
    )
    def rope(X, cos_table, sin_table, rope_perm, sign_mask):
        sign_full = sign_mask.repeat(n_tokens).rearrange(n_tokens, dhead)
        rotated = X.gather(dhead, rope_perm) * sign_full
        return X * cos_table + rotated * sin_table

    result = rope(
        X=X, cos_table=cos_table, sin_table=sin_table,
        rope_perm=rope_perm, sign_mask=sign_mask,
    )

    x_lo = X.arr[..., :dhead_half]
    x_hi = X.arr[..., dhead_half:]
    rotated_arr = jnp.concatenate([-x_hi, x_lo], axis=-1)
    expected = X.arr * cos_table.arr + rotated_arr * sin_table.arr
    assert jnp.allclose(result.arr, expected, atol=1e-5)


def test_swiglu_mlp(reset):
    """
    SwiGLU MLP block: `out = (silu(x @ W_gate) * (x @ W_up)) @ W_down`
    where `silu(x) = x * sigmoid(x)`. Exercises the new `sigmoid`
    primitive in both kernel and spec, plus the chained einsum
    composition through the residual hidden dimension.
    """
    n_tokens = dim("n_tokens", 8)
    d_model = dim("d_model", 16)
    d_ff = dim("d_ff", 32)

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    X = tjax.random.normal(k1, n_tokens, d_model, name="X")
    W_gate = tjax.random.normal(k2, d_model, d_ff, name="W_gate")
    W_up = tjax.random.normal(k3, d_model, d_ff, name="W_up")
    W_down = tjax.random.normal(k4, d_ff, d_model, name="W_down")

    gate = "(X:n_tokens d_model, W_gate:d_model d_ff -> n_tokens d_ff)"
    up = "(X:n_tokens d_model, W_up:d_model d_ff -> n_tokens d_ff)"
    hidden = f"({gate} * sigmoid({gate}) * {up})"

    @tjax.jit(
        spec=f"({hidden}, W_down:d_ff d_model -> n_tokens d_model) -> "
    )
    def swiglu(X, W_gate, W_up, W_down):
        gate = tjax.einsum(
            X, W_gate, "n_tokens d_model, d_model d_ff -> n_tokens d_ff",
        )
        up = tjax.einsum(
            X, W_up, "n_tokens d_model, d_model d_ff -> n_tokens d_ff",
        )
        hidden = gate * tjax.sigmoid(gate) * up
        return tjax.einsum(
            hidden, W_down,
            "n_tokens d_ff, d_ff d_model -> n_tokens d_model",
        )

    result = swiglu(X=X, W_gate=W_gate, W_up=W_up, W_down=W_down)

    # Numerical reference.
    gate_arr = X.arr @ W_gate.arr
    up_arr = X.arr @ W_up.arr
    hidden_arr = gate_arr * jax.nn.sigmoid(gate_arr) * up_arr
    expected = hidden_arr @ W_down.arr
    assert jnp.allclose(result.arr, expected, atol=1e-4)
