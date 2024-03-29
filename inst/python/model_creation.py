import jax
import jax.numpy as jnp
import haiku as hk
#from utils import StateProbFolded, StateProbBound, between, Between, constrained_gradients
from thermo_models import StateProbFolded, StateProbBound
import optax
from chemical_models import create_chemical_model

def create_model_fn(number_additive_traits, l1, l2, rng, model_type = 'tri_state_equilibrium_explicit'):

    def model_fn(inputs_select, inputs_folding, inputs_binding):

        input_layer_select_folding = jnp.expand_dims(inputs_select[:, 0], -1)
        input_layer_select_binding = jnp.expand_dims(inputs_select[:, 1], -1)

        synthesis_additive_trait_layer = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_in",
                                                                                        "truncated_normal"),
                                                 with_bias=False,
                                                 name = 'synthesis_additive_trait'
                                                 )(inputs_folding)

        degradation_additive_trait_layer = hk.Linear(number_additive_traits,
                                            w_init=hk.initializers.VarianceScaling(1.0, "fan_in",
                                                                                "truncated_normal"),
                                            with_bias=False,
                                            name = 'synthesis_additive_trait'
                                            )(inputs_binding)

        module = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_in",
                                                                                        "truncated_normal"),
                                                 with_bias=False,
                                                 name = 'folding_additive_trait'
                                                 )

        folding_additive_trait_layer_foldingset = module(inputs_folding)

        binding_additive_trait_layer = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_in",
                                                                                        "truncated_normal"),
                                                 with_bias=False,
                                                 name = 'binding_additive_trait'
                                                 )(inputs_binding)

        if model_type == 'tri_state_non_equilibrium_implicit' or model_type == 'tri_state_non_equilibrium_ODE':
            folding_nonlinear_layer = StateProbFolded(model_type)(binding_additive_trait_layer,
                                                                  folding_additive_trait_layer_foldingset,
                                                                  synthesis_additive_trait_layer)
        else:
            folding_nonlinear_layer = StateProbFolded(model_type)(binding_additive_trait_layer,
                                                                  folding_additive_trait_layer_foldingset)

        folding_additive_layer = hk.Linear(1,
                                           w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                                           with_bias=True,
                                           name = 'folding_additive'
                                           )(folding_nonlinear_layer)

        if model_type == 'tri_state_non_equilibrium_implicit' or model_type == 'tri_state_non_equilibrium_ODE':
            binding_nonlinear_layer = StateProbBound(model_type)(binding_additive_trait_layer,
                                                                 folding_additive_trait_layer_foldingset,
                                                                 synthesis_additive_trait_layer,
                                                                 degradation = degradation_additive_trait_layer)
        else:
            binding_nonlinear_layer = StateProbBound(model_type)(binding_additive_trait_layer,
                                                                 folding_additive_trait_layer_foldingset,
                                                                 degradation = degradation_additive_trait_layer)

        binding_additive_layer = hk.Linear(1,
                                           w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                                           with_bias=True,
                                           name = 'binding_additive'
                                           )(binding_nonlinear_layer)

        multiplicative_layer_folding = folding_additive_layer * input_layer_select_folding
        multiplicative_layer_binding = binding_additive_layer * input_layer_select_binding
        output_layer = multiplicative_layer_folding + multiplicative_layer_binding

        return output_layer.flatten(), folding_additive_layer, binding_additive_layer, folding_additive_trait_layer_foldingset, binding_additive_trait_layer

    return model_fn

def create_model_fn_class(number_additive_traits, l1, l2, rng, is_implicit = True,
                          model_type = 'tri_state_equilibrium_explicit'):

    def model_fn_class(inputs_select, inputs_folding, inputs_binding):

        model = create_chemical_model(model_type, is_implicit)

        input_layer_select_folding = jnp.expand_dims(inputs_select[:, 0], -1)
        input_layer_select_binding = jnp.expand_dims(inputs_select[:, 1], -1)

        synthesis_additive_trait_layer = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_in",
                                                                                        "truncated_normal"),
                                                 with_bias=False,
                                                 name = 'synthesis_additive_trait'
                                                 )(inputs_folding)

        degradation_additive_trait_layer = hk.Linear(number_additive_traits,
                                            w_init=hk.initializers.VarianceScaling(1.0, "fan_in",
                                                                                "truncated_normal"),
                                            with_bias=False,
                                            name = 'synthesis_additive_trait'
                                            )(inputs_binding)
        #jax.debug.print("{x}",x=inputs_folding.shape)
        folding_additive_trait_layer_foldingset = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_in",
                                                                                        "truncated_normal"),
                                                 with_bias=False,
                                                 name = 'folding_additive_trait'
                                                 )(inputs_folding)

        # binding
        binding_additive_trait_layer = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_in",
                                                                                        "truncated_normal"),
                                                 with_bias=False,
                                                 name = 'binding_additive_trait'
                                                 )(inputs_binding)

        args_folding = (synthesis_additive_trait_layer, folding_additive_trait_layer_foldingset)
        folding_nonlinear_layer = model.solve_folding(args_folding)

        folding_additive_layer = hk.Linear(1,
                                           w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                                           with_bias=True,
                                           name = 'folding_additive'
                                           )(folding_nonlinear_layer)

        args_binding = (synthesis_additive_trait_layer,
                        folding_additive_trait_layer_foldingset,
                        binding_additive_trait_layer,
                        degradation_additive_trait_layer)

        binding_nonlinear_layer = model.solve_binding(args_binding)


        binding_additive_layer = hk.Linear(1,
                                           w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                                           with_bias=True,
                                           name = 'binding_additive'
                                           )(binding_nonlinear_layer)

        multiplicative_layer_folding = folding_additive_layer * input_layer_select_folding
        multiplicative_layer_binding = binding_additive_layer * input_layer_select_binding
        output_layer = multiplicative_layer_folding + multiplicative_layer_binding

        return output_layer.flatten(), folding_additive_layer, binding_additive_layer, folding_additive_trait_layer_foldingset, binding_additive_trait_layer

    return model_fn_class

def create_model_fn_complex(number_additive_traits, l1, l2, rng, is_implicit = True, model_type = 'tri_state_equilibrium_explicit'):

    def model_fn_complex(inputs_select,
                         inputs_mutation_folding,
                         inputs_location_folding,
                         inputs_mutation_binding,
                         inputs_location_binding):

        # Flatten the last two dimensions (feature matrix)
        inputs_mutation_folding = inputs_mutation_folding.reshape((inputs_mutation_folding.shape[0], -1))
        inputs_mutation_binding= inputs_mutation_binding.reshape((inputs_mutation_binding.shape[0], -1))

        #jax.debug.print("{x}", x=inputs_mutation_folding.shape)
        #jax.debug.print("{x}", x=inputs_mutation_folding.shape[1])

        #jax.debug.print("{x}", x=inputs_location_folding[0])
        #jax.debug.print("{x}", x=inputs_location_folding.shape)
        #jax.debug.print("{x}", x=inputs_location_binding[0])
        #jax.debug.print("{x}", x=inputs_mutation_binding[0])

        input_layer_select_folding = jnp.expand_dims(inputs_select[:, 0], -1)
        input_layer_select_binding = jnp.expand_dims(inputs_select[:, 1], -1)

        model = create_chemical_model(model_type, is_implicit)

        ####################### Gibbs free energy #############################
        #fold
        mutation_layer_folding = hk.Linear(number_additive_traits,
                                   w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                                   with_bias=False,
                                   name='mutation_layer_fold')(inputs_mutation_folding)
        location_layer_folding = hk.Linear(number_additive_traits,
                                   w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                                   with_bias=False,
                                   name='location_layer_fold')(inputs_location_folding)

        concat_input_fold = jnp.concatenate((mutation_layer_folding, location_layer_folding), axis=1)
        folding_additive_trait_layer = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_in","truncated_normal"),
                                                 name='folding_additive_trait')(concat_input_fold)

        #bind
        mutation_layer_binding = hk.Linear(number_additive_traits,
                                   w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                                   with_bias=False,
                                   name='mutation_layer_bind')(inputs_mutation_binding)

        location_layer_binding = hk.Linear(number_additive_traits,
                                   w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                                   with_bias=False,
                                   name='location_layer_bind')(inputs_location_binding)
        concat_input_bind = jnp.concatenate((mutation_layer_binding, location_layer_binding), axis=1)
        binding_additive_trait_layer = hk.Linear(number_additive_traits,
                                                 w_init=hk.initializers.VarianceScaling(1.0, "fan_in","truncated_normal"),
                                                 name='binding_additive_trait')(concat_input_bind)

        #synthesis
        mutation_layer_synthesis = hk.Linear(number_additive_traits,
                                   w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                                   with_bias=False,
                                   name='mutation_layer_fold')(inputs_mutation_folding)
        location_layer_synthesis = hk.Linear(number_additive_traits,
                                   w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                                   with_bias=False,
                                   name='location_layer_fold')(inputs_location_folding)
        synthesis_additive_trait_layer = mutation_layer_synthesis + location_layer_synthesis

        #degradation
        mutation_layer_degradation = hk.Linear(number_additive_traits,
                                   w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                                   with_bias=False,
                                   name='mutation_layer_bind')(inputs_mutation_binding)

        location_layer_degradation = hk.Linear(number_additive_traits,
                                   w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal"),
                                   with_bias=False,
                                   name='location_layer_bind')(inputs_location_binding)

        degradation_additive_trait_layer = mutation_layer_degradation + location_layer_degradation

        ####################### Folding and binding #############################
        args_folding = (synthesis_additive_trait_layer, folding_additive_trait_layer)
        folding_nonlinear_layer = model.solve_folding(args_folding)

        folding_additive_layer = hk.Linear(1,
                                           w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                                           with_bias=True,
                                           name = 'folding_additive'
                                           )(folding_nonlinear_layer)

        args_binding = (synthesis_additive_trait_layer,
                        folding_additive_trait_layer,
                        binding_additive_trait_layer,
                        degradation_additive_trait_layer)
        binding_nonlinear_layer = model.solve_binding(args_binding)

        binding_additive_layer = hk.Linear(1,
                                           w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                                           with_bias=True,
                                           name = 'binding_additive'
                                           )(binding_nonlinear_layer)

        ####################### OUTPUT LAYERS
        multiplicative_layer_folding = folding_additive_layer * input_layer_select_folding
        multiplicative_layer_binding = binding_additive_layer * input_layer_select_binding
        output_layer = multiplicative_layer_folding + multiplicative_layer_binding

        return output_layer.flatten(), folding_additive_layer, binding_additive_layer, folding_additive_trait_layer, binding_additive_trait_layer

    return model_fn_complex



def create_model_jax(rng, learn_rate, l1, l2,
                     number_additive_traits, model_type = 'tri_state_explicit', is_implicit = True, is_complex = False):

    if is_complex == True:
        model_fn = create_model_fn_complex(number_additive_traits, l1, l2, rng, is_implicit, model_type)

    elif is_complex == False:
        model_fn = create_model_fn_class(number_additive_traits, l1, l2, rng, is_implicit, model_type)
        #model_fn = create_model_fn(number_additive_traits, l1, l2, rng, model_type)

    model = hk.without_apply_rng(hk.transform(model_fn))

    opt_init, opt_update = optax.adam(learn_rate)

    return model, opt_init, opt_update
