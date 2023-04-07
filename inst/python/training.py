import jax
import jax.numpy as jnp
import optax
from itertools import zip_longest
from model_creation import create_model_jax

def shuffle_weights(rng, params):
    def _shuffle_array(rng, arr):
        flat_arr = arr.ravel()
        shuffled_flat_arr = jax.random.permutation(rng, flat_arr)
        return jnp.reshape(shuffled_flat_arr, arr.shape)

    leaves, _ = jax.tree_util.tree_flatten(params)
    rngs = jax.random.split(rng, len(leaves))
    zipped_args = zip_longest(leaves, rngs, fillvalue=None)
    new_leaves = [(_shuffle_array(rng, leaf) if leaf is not None else leaf) for leaf, rng in zipped_args]
    new_params = jax.tree_util.tree_unflatten(_, new_leaves)

    return new_params

def generate_batches(input_data, batch_size, rng):
    """Generate batches for training.

    Args:
        input_data: A dictionary of NumPy arrays containing the input data.
        batch_size: The batch size.
        rng: A JAX PRNGKey.

    Yields:
        A tuple of (select, fold, bind, target) batches.
    """
    num_samples = input_data['select'].shape[0]
    indices = jnp.arange(num_samples)

    # Shuffle the training data.
    rng, _ = jax.random.split(rng)
    indices = jax.random.permutation(rng, indices)

    # Generate batches.
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_select = input_data['select'][batch_indices]
        batch_fold = input_data['fold'][batch_indices]
        batch_bind = input_data['bind'][batch_indices]
        batch_target = input_data['target'][batch_indices]

        yield batch_select, batch_fold, batch_bind, batch_target


def model_training(model, optimizer, weights, opt_state, param_dict, input_data, n_epochs, rng):
    print("Grid search using %s" % (param_dict))

    rng_init, rng_batches = jax.random.split(rng)

    @jax.jit
    def loss_fn(params, inputs_select, inputs_folding, inputs_binding, target):
        output, folding_additive_trait_layer, binding_additive_trait_layer = model.apply(params, inputs_select, inputs_folding, inputs_binding)
        loss = jnp.mean(jnp.abs(output - target))
        return loss

    @jax.jit
    def update(weights, opt_state, inputs_select, inputs_folding, inputs_binding, target):
        grads = jax.grad(loss_fn)(weights, inputs_select, inputs_folding, inputs_binding, target)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return weights, new_opt_state

    history = []
    for epoch in range(n_epochs):

        for batch_data in generate_batches(input_data['train'], param_dict['num_samples'], rng_batches):
            inputs_select, inputs_folding, inputs_binding, target = batch_data
            weights, opt_state = update(weights, opt_state, inputs_select, inputs_folding, inputs_binding, target)

        val_loss = loss_fn(weights, input_data['valid']['select'], input_data['valid']['fold'],
                           input_data['valid']['bind'], input_data['valid']['target'])
        history.append(val_loss.item())
        print('epoch done')

    return history, model, weights

def generate_batches(input_data, batch_size, rng):
    """Generate batches for training.

    Args:
        input_data: A dictionary of NumPy arrays containing the input data.
        batch_size: The batch size.
        rng: A JAX PRNGKey.

    Yields:
        A tuple of (select, fold, bind, target) batches.
    """
    num_samples = input_data['select'].shape[0]
    indices = jnp.arange(num_samples)

    # Shuffle the training data.
    rng, _ = jax.random.split(rng)
    indices = jax.random.permutation(rng, indices)

    # Generate batches.
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_select = input_data['select'][batch_indices]
        batch_fold = input_data['fold'][batch_indices]
        batch_bind = input_data['bind'][batch_indices]
        batch_target = input_data['target'][batch_indices]

        yield batch_select, batch_fold, batch_bind, batch_target

def fit_model_grid_jax(param_dict, input_data, n_epochs, rng):
    # Summarize results
    print("Grid search using %s" % (param_dict))

    rng_init, rng_batches = jax.random.split(rng)

    # Create model
    model, optimizer = create_model_jax(
        rng=rng_init,
        learn_rate=param_dict['learning_rate'],
        l1=param_dict['l1_regularization_factor'],
        l2=param_dict['l2_regularization_factor'],
        input_dim_select=input_data['train']['select'].shape[1],
        input_dim_folding=input_data['train']['fold'].shape[1],
        input_dim_binding=input_data['train']['bind'].shape[1],
        number_additive_traits=param_dict['number_additive_traits'])

    @jax.jit
    def loss_fn(params, inputs_select, inputs_folding, inputs_binding, target):
        output, folding_additive_trait_layer, binding_additive_trait_layer = model.apply(params, inputs_select, inputs_folding, inputs_binding)
        loss = jnp.mean(jnp.abs(output - target))

        # Apply L1 and L2 regularization
        l1_loss = 0
        l2_loss = 0
        for p in jax.tree_util.tree_leaves(params):
            if p.ndim > 1:  # exclude bias parameters
                l1_loss += jnp.sum(jnp.abs(p))
                l2_loss += jnp.sum(jnp.square(p))
        loss = loss + param_dict['l1_regularization_factor'] * l1_loss + param_dict[
            'l2_regularization_factor'] * l2_loss

        return loss

    @jax.jit
    def update(params, opt_state, inputs_select, inputs_folding, inputs_binding, target):
        grads = jax.grad(loss_fn)(params, inputs_select, inputs_folding, inputs_binding, target)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        # print('update done')
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    params = model.init(rng, input_data['train']['select'], input_data['train']['fold'], input_data['train']['bind'])
    opt_state = optimizer.init(params)

    for epoch in range(n_epochs):
        for batch_data in generate_batches(input_data['train'], param_dict['num_samples'], rng_batches):
            inputs_select, inputs_folding, inputs_binding, target = batch_data
            params, opt_state = update(params, opt_state, inputs_select, inputs_folding, inputs_binding, target)
        val_loss = loss_fn(params, input_data['valid']['select'], input_data['valid']['fold'],
                           input_data['valid']['bind'], input_data['valid']['target'])
        print('epoch done')

    return val_loss.item()
