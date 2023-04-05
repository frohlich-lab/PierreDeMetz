import jax
import jax.numpy as jnp
from jax import random, jit, lax
from typing import Optional, Callable
import numpy as np

def custom_linear(
    inputs: jax.Array,
    output_size: int,
    with_bias: bool = True,
    w_init: Optional[Callable] = None,
    b_init: Optional[Callable] = None,
    rng_key: Optional[jax.numpy.ndarray] = None,
    precision: Optional[lax.Precision] = None,
) -> jax.Array:

    input_size = inputs.shape[-1]
    dtype = inputs.dtype

    if rng_key is None:
        rng_key = random.PRNGKey(0)

    if w_init is None:
        stddev = 1. / np.sqrt(input_size)
        w_init = jax.nn.initializers.glorot_normal()

    if b_init is None:
        b_init = jax.nn.initializers.zeros
    
    w_key, b_key = random.split(rng_key, 2)
    w = w_init(w_key, (input_size, output_size), dtype)
    out = jnp.dot(inputs, w, precision=precision)

    if with_bias:
        b = b_init(b_key, (output_size,), dtype)
        b = jnp.broadcast_to(b, out.shape)
        out = out + b

    return out


# You can also create a JIT-compiled version of the custom_linear function
custom_linear_jit = jit(custom_linear, static_argnums=(1,))