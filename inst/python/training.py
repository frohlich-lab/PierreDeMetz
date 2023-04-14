import jax
import jax.numpy as jnp
import optax
from model_creation import create_model_jax
import numpy as np
from utils import apply_weight_constraints


def generate_batches(input_data, batch_size, rng):

    num_samples = input_data['select'].shape[0]
    indices = jnp.arange(num_samples)
    indices = jax.random.permutation(rng, indices)

    for start_idx in range(0, num_samples, batch_size):

        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_select = input_data['select'][batch_indices]
        batch_fold = input_data['fold'][batch_indices]
        batch_bind = input_data['bind'][batch_indices]
        batch_target = input_data['target'][batch_indices]

        yield batch_select, batch_fold, batch_bind, batch_target


def model_training(model, opt_state,opt_update, weights, param_dict, input_data, n_epochs, rng):

    print("Training the model with %s" % (param_dict))
    rng_batches = jax.random.split(rng, num=n_epochs)

    @jax.jit
    def loss_fn(weights, inputs_select, inputs_folding, inputs_binding, target):
        output, _, _, _, _  = model.apply(weights, inputs_select, inputs_folding, inputs_binding)
        #jax.debug.print('weights : {}', weights['folding_additive'])
        #jax.debug.print('weights : {}', weights['folding_additive_trait'])
        #jax.debug.print('target : {}', target[:10])
        #jax.debug.print('output : {}', output[:10])
        loss = jnp.mean(jnp.abs(target-output))
        return loss

    @jax.jit
    def update(weights, opt_state, inputs_select, inputs_folding, inputs_binding, target):
        loss, grads = jax.value_and_grad(loss_fn)(weights, inputs_select, inputs_folding, inputs_binding, target)
        updates, opt_state = opt_update(grads, opt_state)
        #jax.debug.print('weights : {}', weights['folding_additive'])
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state

    history = []
    for epoch in range(n_epochs):
        batch_data = generate_batches(input_data['train'], param_dict['num_samples'], rng_batches[epoch])
        for batch in batch_data:

            inputs_select, inputs_folding, inputs_binding, target = batch
            weights, opt_state = update(weights, opt_state, inputs_select, inputs_folding, inputs_binding, target)

            weights = apply_weight_constraints(weights, 'folding_additive', 0, 1e3)
            weights = apply_weight_constraints(weights, 'binding_additive', 0, 1e3)

        val_loss = loss_fn(weights, input_data['valid']['select'], input_data['valid']['fold'],
                           input_data['valid']['bind'], input_data['valid']['target'])

        history.append(val_loss.item())
        print(f'epoch done with {val_loss.item()}')

    return history, model, weights





def fit_model_grid_jax(param_dict, input_data, n_epochs, rng):
    # Summarize results
    print("Grid search using %s" % (param_dict))

    rng_batches = jax.random.split(rng, num=n_epochs)

    # Create model
    model, optimizer = create_model_jax(
        rng=rng,
        learn_rate=param_dict['learning_rate'],
        l1=param_dict['l1_regularization_factor'],
        l2=param_dict['l2_regularization_factor'],
        input_dim_select=input_data['train']['select'].shape[1],
        input_dim_folding=input_data['train']['fold'].shape[1],
        input_dim_binding=input_data['train']['bind'].shape[1],
        number_additive_traits=param_dict['number_additive_traits'])

    #@jax.jit
    def loss_fn(params, inputs_select, inputs_folding, inputs_binding, target):

        output, folding_additive_layer, binding_additive_layer, folding_additive_trait_layer, binding_additive_trait_layer = model.apply(params, inputs_select, inputs_folding, inputs_binding)
        loss = jnp.mean(jnp.abs(output - target))

        # Apply L1 and L2 regularization
        l1_loss = 0
        l2_loss = 0
        binding_additive_trait_params = params['binding_additive_trait']['w']

        if binding_additive_trait_params.ndim > 1:  # exclude bias parameters
            l1_loss += jnp.sum(jnp.abs(binding_additive_trait_params))
            l2_loss += jnp.sum(jnp.square(binding_additive_trait_params))

        loss = loss + param_dict['l1_regularization_factor'] * l1_loss + param_dict['l2_regularization_factor'] * l2_loss
        return loss

    #@jax.jit
    def update(params, opt_state, inputs_select, inputs_folding, inputs_binding, target):

        grads = jax.grad(loss_fn)(params, inputs_select, inputs_folding, inputs_binding, target)

        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    params = model.init(rng, input_data['train']['select'], input_data['train']['fold'], input_data['train']['bind'])
    opt_state = optimizer.init(params)

    for epoch in range(n_epochs):
        #print(params['binding_additive'])
        #print(params['folding_additive'])

        for batch_data in generate_batches(input_data['train'], param_dict['num_samples'], rng_batches[epoch]):
            inputs_select, inputs_folding, inputs_binding, target = batch_data
            params, opt_state = update(params, opt_state, inputs_select, inputs_folding, inputs_binding, target)

            params = apply_weight_constraints(params, 'folding_additive', 0, 1e3)
            params = apply_weight_constraints(params, 'binding_additive', 0, 1e3)


        val_loss = loss_fn(params, input_data['valid']['select'], input_data['valid']['fold'],
                           input_data['valid']['bind'], input_data['valid']['target'])

        print(f'epoch done with {val_loss.item()}')

    return val_loss.item()
