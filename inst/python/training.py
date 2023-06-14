import jax
import jax.numpy as jnp
import optax
from model_creation import create_model_jax
import numpy as np
from utils import apply_weight_constraints
import wandb


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

def generate_batches_complex(input_data, batch_size, rng):
    num_samples = input_data['select'].shape[0]
    indices = jnp.arange(num_samples)
    indices = jax.random.permutation(rng, indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_select = input_data['select'][batch_indices]
        batch_fold_mutation = input_data['fold_mutation'][batch_indices]
        batch_fold_location = input_data['fold_location'][batch_indices]
        batch_bind_mutation = input_data['bind_mutation'][batch_indices]
        batch_bind_location = input_data['bind_location'][batch_indices]
        batch_target = input_data['target'][batch_indices]

        yield batch_select, batch_fold_mutation, batch_fold_location, batch_bind_mutation, batch_bind_location, batch_target


def model_training(model, opt_state,opt_update, weights, param_dict, input_data, n_epochs, rng, wandb_config):

    print("Training the model with %s" % (param_dict))
    rng_batches = jax.random.split(rng, num=n_epochs)
    wandb_config_updated = {**wandb_config, **param_dict}

    if wandb_config['status'] == 'True':
        wandb.init(
        project=wandb_config['project_name'],
        entity = 'lab_frohlich',
        config=wandb_config_updated,
        group = wandb_config['model_type'],
        job_type='final_model_training',
        reinit=True,
        name = f'Actual model training : {param_dict}')

    @jax.jit
    def loss_fn(weights, inputs_select, inputs_folding, inputs_binding, target):
        output, _, _, _, _  = model.apply(weights, inputs_select, inputs_folding, inputs_binding)
        loss = jnp.mean(jnp.abs(target-output))
        return loss

    @jax.jit
    def update(weights, opt_state, inputs_select, inputs_folding, inputs_binding, target):
        loss, grads = jax.value_and_grad(loss_fn)(weights, inputs_select, inputs_folding, inputs_binding, target)
        updates, opt_state = opt_update(grads, opt_state)
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
        print(f'epoch done with {val_loss.item():.3f}')

        if wandb_config['status'] == 'True':
            wandb.log({'val_loss_train': round(val_loss.item(),3)})

    return history, model, weights





def fit_model_grid_jax(param_dict, input_data, n_epochs, rng, wandb_config):

    print("Grid search using %s" % (param_dict))

    rng_batches = jax.random.split(rng, num=n_epochs)
    run_number = wandb_config.get('run_number', 1)
    wandb_config_updated = {**wandb_config, **param_dict}

    if wandb_config_updated['status'] == 'True':
        wandb.init(
            project=wandb_config['project_name'],
            entity = 'lab_frohlich',
            config=wandb_config_updated,
            group=wandb_config['model_type'],
            job_type='grid_search',
            reinit=True,
            name=f'Run {run_number}'
        )

    # Create model
    model, opt_init, opt_update = create_model_jax(
        rng=rng,
        learn_rate=param_dict['learning_rate'],
        l1=param_dict['l1_regularization_factor'],
        l2=param_dict['l2_regularization_factor'],
        number_additive_traits=param_dict['number_additive_traits'],
        model_type=param_dict['model_type'],
        is_implicit=param_dict['is_implicit'],
        is_complex=param_dict['is_complex']
        )

    @jax.jit
    def loss_fn(weights, inputs_select, inputs_folding, inputs_binding, target):
        output, _, _, _, _  = model.apply(weights, inputs_select, inputs_folding, inputs_binding)
        loss = jnp.mean(jnp.abs(target-output))
        return loss

    @jax.jit
    def update(weights, opt_state, inputs_select, inputs_folding, inputs_binding, target):
        grads = jax.grad(loss_fn)(weights, inputs_select, inputs_folding, inputs_binding, target)
        updates, opt_state = opt_update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state

    weights = model.init(rng,
                        jnp.ones_like(input_data['train']['select']),
                        jnp.ones_like(input_data['train']['fold']),
                        jnp.ones_like(input_data['train']['bind']))
    opt_state = opt_init(weights)
    weights = apply_weight_constraints(weights, 'folding_additive', 0, 1e3)
    weights = apply_weight_constraints(weights, 'binding_additive', 0, 1e3)


    history = []
    for epoch in range(n_epochs):
        batch_data = generate_batches(input_data['train'], param_dict['num_samples'], rng_batches[epoch])
        for batch in batch_data:

            inputs_select, inputs_folding, inputs_binding, target = batch
            weights, opt_state = update(weights, opt_state, inputs_select, inputs_folding, inputs_binding, target)
            #print(weights['folding_additive_trait']['w'])

            weights = apply_weight_constraints(weights, 'folding_additive', 0, 1e3)
            weights = apply_weight_constraints(weights, 'binding_additive', 0, 1e3)

        val_loss = loss_fn(weights, input_data['valid']['select'], input_data['valid']['fold'],
                           input_data['valid']['bind'], input_data['valid']['target'])

        history.append(val_loss.item())
        print(f'epoch done with {val_loss.item():.3f}')

        if wandb_config['status'] == 'True':
            wandb.log({'val_loss_fit': round(val_loss.item(),3)})

    return val_loss.item()


def fit_model_grid_complex(param_dict, input_data, n_epochs, rng, wandb_config):

    print("Grid search using %s" % (param_dict))

    rng_batches = jax.random.split(rng, num=n_epochs)
    run_number = wandb_config.get('run_number', 1)
    wandb_config_updated = {**wandb_config, **param_dict}

    if wandb_config_updated['status'] == 'True':
        wandb.init(
            project=wandb_config['project_name'],
            entity = 'lab_frohlich',
            config=wandb_config_updated,
            group=wandb_config['model_type'],
            job_type='grid_search',
            reinit=True,
            name=f'Run {run_number}'
        )

    # Create model
    # Update model creation code to pass correct input sizes
    model, opt_init, opt_update = create_model_jax(
        rng=rng,
        learn_rate=param_dict['learning_rate'],
        l1=param_dict['l1_regularization_factor'],
        l2=param_dict['l2_regularization_factor'],
        number_additive_traits=param_dict['number_additive_traits'],
        model_type=param_dict['model_type'],
        is_implicit=param_dict['is_implicit'],
        is_complex=param_dict['is_complex']
    )

    @jax.jit
    def loss_fn(weights, inputs_select, inputs_folding_mutation, inputs_folding_location,
                inputs_binding_mutation, inputs_binding_location, target):
        output, _, _, _, _ = model.apply(weights, inputs_select, inputs_folding_mutation, inputs_folding_location,
                                         inputs_binding_mutation, inputs_binding_location)
        loss = jnp.mean(jnp.abs(target - output))
        return loss

    @jax.jit
    def update(weights, opt_state, inputs_select, inputs_folding_mutation, inputs_folding_location,
                inputs_binding_mutation, inputs_binding_location, target):
        grads = jax.grad(loss_fn)(weights, inputs_select, inputs_folding_mutation, inputs_folding_location,
                                  inputs_binding_mutation, inputs_binding_location, target)
        updates, opt_state = opt_update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state

    weights = model.init(rng,
                         jnp.ones_like(input_data['train']['select']),
                         jnp.ones_like(input_data['train']['fold_mutation']),
                         jnp.ones_like(input_data['train']['fold_location']),
                         jnp.ones_like(input_data['train']['bind_mutation']),
                         jnp.ones_like(input_data['train']['bind_location']))

    opt_state = opt_init(weights)
    weights = apply_weight_constraints(weights, 'folding_additive', 0, 1e3)
    weights = apply_weight_constraints(weights, 'binding_additive', 0, 1e3)

    history = []
    for epoch in range(n_epochs):
        batch_data = generate_batches_complex(input_data['train'], param_dict['num_samples'], rng_batches[epoch])
        for batch in batch_data:
            inputs_select, inputs_folding_mutation, inputs_folding_location, inputs_binding_mutation, inputs_binding_location, target = batch
            weights, opt_state = update(weights, opt_state, inputs_select, inputs_folding_mutation, inputs_folding_location,
                                        inputs_binding_mutation, inputs_binding_location, target)
            #print(weights['mutation_layer_fold'])
            #print(weights['folding_additive_trait']['w'])

            weights = apply_weight_constraints(weights, 'folding_additive', 0, 1e3)
            weights = apply_weight_constraints(weights, 'binding_additive', 0, 1e3)

        val_loss = loss_fn(weights, input_data['valid']['select'],
                        input_data['valid']['fold_mutation'],
                        input_data['valid']['fold_location'],
                        input_data['valid']['bind_mutation'],
                        input_data['valid']['bind_location'],
                        input_data['valid']['target'])

        history.append(val_loss.item())
        print(f'epoch done with {val_loss.item():.3f}')

        if wandb_config['status'] == 'True':
            wandb.log({'val_loss_fit': round(val_loss.item(),3)})

    return val_loss.item()



def model_training_complex(model, opt_state,opt_update, weights, param_dict, input_data, n_epochs, rng, wandb_config):

    print("Training the model with %s" % (param_dict))
    rng_batches = jax.random.split(rng, num=n_epochs)
    wandb_config_updated = {**wandb_config, **param_dict}

    if wandb_config['status'] == 'True':
        wandb.init(
        project=wandb_config['project_name'],
        entity = 'lab_frohlich',
        config=wandb_config_updated,
        group = wandb_config['model_type'],
        job_type='final_model_training',
        reinit=True,
        name = f'Actual model training : {param_dict}')

    @jax.jit
    def loss_fn(weights, inputs_select, inputs_folding_mutation, inputs_folding_location,
                inputs_binding_mutation, inputs_binding_location, target):
        output, _, _, _, _ = model.apply(weights, inputs_select, inputs_folding_mutation, inputs_folding_location,
                                         inputs_binding_mutation, inputs_binding_location)
        loss = jnp.mean(jnp.abs(target - output))
        return loss

    @jax.jit
    def update(weights, opt_state, inputs_select, inputs_folding_mutation, inputs_folding_location,
                inputs_binding_mutation, inputs_binding_location, target):
        grads = jax.grad(loss_fn)(weights, inputs_select, inputs_folding_mutation, inputs_folding_location,
                                  inputs_binding_mutation, inputs_binding_location, target)
        updates, opt_state = opt_update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        return weights, opt_state

    history = []
    for epoch in range(n_epochs):
        batch_data = generate_batches_complex(input_data['train'], param_dict['num_samples'], rng_batches[epoch])
        for batch in batch_data:
            inputs_select, inputs_folding_mutation, inputs_folding_location, inputs_binding_mutation, inputs_binding_location, target = batch
            weights, opt_state = update(weights, opt_state, inputs_select, inputs_folding_mutation, inputs_folding_location,
                                        inputs_binding_mutation, inputs_binding_location, target)
            weights = apply_weight_constraints(weights, 'folding_additive', 0, 1e3)
            weights = apply_weight_constraints(weights, 'binding_additive', 0, 1e3)

        val_loss = loss_fn(weights, input_data['valid']['select'],
                        input_data['valid']['fold_mutation'],
                        input_data['valid']['fold_location'],
                        input_data['valid']['bind_mutation'],
                        input_data['valid']['bind_location'],
                        input_data['valid']['target'])

        history.append(val_loss.item())
        print(f'epoch done with {val_loss.item():.3f}')

        if wandb_config['status'] == 'True':
            wandb.log({'val_loss_train': round(val_loss.item(),3)})

    return history, model, weights
