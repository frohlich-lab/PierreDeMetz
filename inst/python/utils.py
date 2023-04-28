import jax
import jax.nn as nn
from jax.random import normal
from jax.experimental import sparse
#from sparse import bcoo_concatenate
import jax.numpy as jnp
import haiku as hk
import optax
from jax import jit
from chem_model import opt_soln_tri_state, opt_soln_two_state
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


############BASIC THERMO MODEL################
class StateProbFolded(hk.Module):
    def __call__(self, inputs):
        return 1/(1+jnp.exp(inputs))

class StateProbBound(hk.Module):
    def __call__(self, inputs_1, inputs_2):
        return 1/(1+jnp.exp(inputs_1)*(1+jnp.exp(inputs_2)))

#############IMPLICIT THERMO MODEL#############
class StateProbBound_Implicit(hk.Module):
    def __call__(self, inputs_1, inputs_2):
        return opt_soln_tri_state(inputs_1, inputs_2)

class StateProbFolded_Explicit(hk.Module):
    def __call__(self, inputs_1):
        return opt_soln_two_state(inputs_1)

#############MISCELLANEOUS#############
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

def get_layer_index(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx

def get_seq_id(sq):
    return ":".join([str(i)+sq[i] for i in range(len(sq))])


def apply_weight_constraints(params, layer_name, min_val, max_val):
    constrained_params = params.copy()
    constrained_params[layer_name]['w'] = jnp.clip(constrained_params[layer_name]['w'], min_val, max_val)
    return constrained_params

def shuffle_weights(new_rng, weights):
    updated_weights = {}
    for layer, value in weights.items():
        layer_weights = {}
        for key, weight in value.items():
            if key == 'w':
                shuffled_weight = jax.random.permutation(new_rng, weight)
                layer_weights[key] = shuffled_weight
            else:
                layer_weights[key] = weight
        updated_weights[layer] = layer_weights

    return updated_weights
