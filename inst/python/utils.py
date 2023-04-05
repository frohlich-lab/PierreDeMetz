import jax
import jax.nn as nn
from jax.random import normal
from jax.experimental import sparse
#from sparse import bcoo_concatenate
from jax.tree_util import tree_map
import jaxlib
import jax.numpy as jnp
import haiku as hk
import optax
from jax import jit
from functools import partial
import functools

from optax import GradientTransformation

def constrained_gradients(layer_names, min_value, max_value) -> GradientTransformation:
    def init_fn(_):
        return ()

    def update_fn(grads, state, _params):
        def clip_grads(g, path):
            return jnp.clip(g, min_value, max_value) if any(name in path for name in layer_names) else g

        constrained_grads = jax.tree_map(clip_grads, grads, jax.tree_map(lambda _: (), grads), is_leaf=lambda g: isinstance(g, jnp.ndarray))
        return constrained_grads, state

    return optax.GradientTransformation(init=init_fn, update=update_fn)


class StateProbFolded(hk.Module):
    def __call__(self, inputs):
        return nn.sigmoid(inputs)

class StateProbBound(hk.Module):
    def __call__(self, inputs_1, inputs_2):
        return nn.sigmoid(inputs_1 + nn.softplus(inputs_2))

class Between(hk.Module):
    def __init__(self, min_value, max_value, name=None):
        super().__init__(name=name)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, inputs):
        return jax.lax.clamp(self.min_value, inputs, self.max_value)
    
def between(min_value, max_value):
    def clip_fn(params):
        return jax.tree_map(lambda w: jnp.clip(w, min_value, max_value), params)
    return clip_fn