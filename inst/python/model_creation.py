import jax
import jax.numpy as jnp
import haiku as hk
#from utils import StateProbFolded, StateProbBound, between, Between, constrained_gradients
from thermo_models import StateProbFolded, StateProbBound
import optax

def create_model_fn(number_additive_traits, l1, l2, rng, model_type = 'tri_state_explicit'):

    def model_fn(inputs_select, inputs_folding, inputs_binding):

        input_layer_select_folding = jnp.expand_dims(inputs_select[:, 0], -1)
        input_layer_select_binding = jnp.expand_dims(inputs_select[:, 1], -1)
        #jax.debug.print('input_folding : {}', inputs_folding.shape)
        #jax.debug.print('input_binding : {}', inputs_binding.shape)

        #what input do I need to give to this function?
        synthesis_additive_trait_layer = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_in",
                                                                                        "truncated_normal"),
                                                 with_bias=False,
                                                 name = 'synthesis_additive_trait'
                                                 )(inputs_folding)
        module = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_in",
                                                                                        "truncated_normal"),
                                                 with_bias=False,
                                                 name = 'folding_additive_trait'
                                                 )

        folding_additive_trait_layer_foldingset = module(inputs_folding)

        #folding_additive_trait_layer_bindingset = module(inputs_binding)

        # binding
        binding_additive_trait_layer = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_in",
                                                                                        "truncated_normal"),
                                                 with_bias=False,
                                                 name = 'binding_additive_trait'
                                                 )(inputs_binding)

        if model_type == 'tri_state_non_equilibrium_implicit':
            folding_nonlinear_layer = StateProbFolded(model_type)(binding_additive_trait_layer, folding_additive_trait_layer_foldingset, synthesis_additive_trait_layer)
        else:
            folding_nonlinear_layer = StateProbFolded(model_type)(binding_additive_trait_layer, folding_additive_trait_layer_foldingset)


        folding_additive_layer = hk.Linear(1,
                                           w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                                           with_bias=True,
                                           name = 'folding_additive'
                                           )(folding_nonlinear_layer)

        # IF WANTING TO USE USION MODE THEN UNCOMMENT THE FOLDING BINDING SET LAYER AND CHANGE FOLDING TO BINDING JUST BELOW
        if model_type == 'tri_state_non_equilibrium_implicit':
            binding_nonlinear_layer = StateProbBound(model_type)(binding_additive_trait_layer, folding_additive_trait_layer_foldingset, synthesis_additive_trait_layer)
        else:
            binding_nonlinear_layer = StateProbBound(model_type)(binding_additive_trait_layer, folding_additive_trait_layer_foldingset)

        binding_additive_layer = hk.Linear(1,
                                           w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                                           with_bias=True,
                                           name = 'binding_additive'
                                           )(binding_nonlinear_layer)

        # output
        multiplicative_layer_folding = folding_additive_layer * input_layer_select_folding
        multiplicative_layer_binding = binding_additive_layer * input_layer_select_binding
        output_layer = multiplicative_layer_folding + multiplicative_layer_binding

        return output_layer.flatten(), folding_additive_layer, binding_additive_layer, folding_additive_trait_layer_foldingset, binding_additive_trait_layer

    return model_fn


def create_model_jax(rng, learn_rate, l1, l2, input_dim_select, input_dim_folding, input_dim_binding,
                     number_additive_traits, model_type = 'tri_state_explicit'):
    # Create model
    model_fn = create_model_fn(number_additive_traits, l1, l2, rng, model_type)
    model = hk.without_apply_rng(hk.transform(model_fn))

    opt_init, opt_update = optax.adam(learn_rate)
    #opt_init, opt_update = optax.sgd(learn_rate)

    # Create optimizer
    #opt = optax.chain(
    #    optax.adam(learn_rate)#,
        #constrained_gradients(['folding_additive', 'binding_additive'], 0, 1e3),
    #)

    return model, opt_init, opt_update
