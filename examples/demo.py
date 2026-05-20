import jax
import jax.numpy as jnp

import stile.jax as tjax # tnp stands for Typed Jax

from stile import dim

M, N = dim('M', 10), dim('N', 10)
key = jax.random.PRNGKey(0)
a = tjax.random.normal(key, M, N)
b = tjax.random.normal(key, M, N)

a_slice = 2 * a.slice(M, 0, 5)
b_slice = b.slice(M, 0, 5)
c_slice = a_slice + b_slice

result = tjax.TypedResult("2 * M N + M N")
result.assign(c_slice)


