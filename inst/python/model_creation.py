import jax
import jax.numpy as jnp
import haiku as hk
from utils import StateProbFolded, StateProbBound, between, Between, constrained_gradients
import optax

def create_model_fn(number_additive_traits, l1, l2, rng):

    def model_fn(inputs_select, inputs_folding, inputs_binding):

        input_layer_select_folding = jnp.expand_dims(inputs_select[:, 0], -1)
        input_layer_select_binding = jnp.expand_dims(inputs_select[:, 1], -1)

        folding_additive_trait_layer = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_avg",
                                                                                        "truncated_normal"),
                                                 with_bias=True,
                                                 name = 'folding_additive_trait'
                                                 )(inputs_folding)

        folding_nonlinear_layer = StateProbFolded()(folding_additive_trait_layer)

        folding_additive_layer = hk.Linear(number_additive_traits,
                                           w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
                                           with_bias=True,
                                           name = 'folding_additive' # ,
                                           # kernel_regularizer=hk.regularizers.L1L2(l1=l1, l2=l2)
                                           )(folding_nonlinear_layer)

        # binding
        binding_additive_trait_layer = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_avg",
                                                                                        "truncated_normal"),
                                                 with_bias=True,
                                                 name = 'binding_additive_trait'
                                                 )(inputs_binding)

        binding_nonlinear_layer = StateProbBound()(binding_additive_trait_layer, folding_additive_trait_layer)

        binding_additive_layer = hk.Linear(number_additive_traits,
                                           w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"),
                                           with_bias=True,
                                           name = 'binding_additive'
                                           )(binding_nonlinear_layer)

        # output
        multiplicative_layer_folding = folding_additive_layer * input_layer_select_folding
        multiplicative_layer_binding = binding_additive_layer * input_layer_select_binding
        output_layer = multiplicative_layer_folding + multiplicative_layer_binding

        return output_layer, folding_additive_trait_layer, binding_additive_trait_layer

    return model_fn


def create_model_jax(rng, learn_rate, l1, l2, input_dim_select, input_dim_folding, input_dim_binding,
                     number_additive_traits):
    # Create model
    model_fn = create_model_fn(number_additive_traits, l1, l2, rng)
    model = hk.without_apply_rng(hk.transform(model_fn))

    # Create optimizer
    opt = optax.chain(
        optax.adam(learn_rate),
        constrained_gradients(['folding_additive', 'binding_additive'], 0, 1e3),
    )

    # Create regularizer
    # regularizer = hk.regularizers.L1L2(l1=l1, l2=l2)

    return model, opt
