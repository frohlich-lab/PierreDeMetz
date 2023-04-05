#!/usr/bin/env python

#######################################################################
## COMMANDLINE ARGUMENTS ##
#######################################################################
import argparse

#Create parser
parser = argparse.ArgumentParser()

#Add arguments to the parser
parser.add_argument("--data_train", help = "Training data")
parser.add_argument("--data_valid", default = None, help = "Validation data")
parser.add_argument("--data_obs", help = "All observed data")
parser.add_argument("--output_directory", "-o")
parser.add_argument("--number_additive_traits", "-n", default = 1, type = int, help = "Number of additive traits")
parser.add_argument("--l1_regularization_factor", default = "0.0001,0.001,0.01,0.1", help = "L1 regularization factor for binding additive trait layer (default:0.0001,0.001,0.01,0.1)")
parser.add_argument("--l2_regularization_factor", default = "0.0001,0.001,0.01,0.1", help = "L2 regularization factor for binding additive trait layer (default:0.0001,0.001,0.01,0.1)")
parser.add_argument("--num_epochs_grid", "-e", default = 100, type = int, help = "Number of epochs to train the model during grid search")
parser.add_argument("--num_epochs", "-p", default = 10000, type = int, help = "Maximum number of epochs to train the final model")
parser.add_argument("--num_samples", "-s", default = "128,256,512,1024", help = "Number of samples per gradient update (default:128,256,512,1024)")
parser.add_argument("--learning_rate", "-a", default = "0.0001,0.001,0.01,0.1", help = "Learning rate (default:0.0001,0.001,0.01,0.1)")
parser.add_argument("--num_resamplings", "-r", default = 10, type = int, help = "Number of random resamples from fitness distribution (default:10)")
parser.add_argument("--early_stopping", "-l", default = False, type = bool, help = "Whether to stop early (default:False)")
parser.add_argument("--num_models", "-u", default = 10, type = int, help = "Number of final models to fit (default:10)")
parser.add_argument("--random_seed", "-d", default = 1, type = int, help = "Random seed (default:1)")

#Parse the arguments
args = parser.parse_args()
#print(args)

data_train_file = args.data_train
data_valid_file = args.data_valid
data_obs_file = args.data_obs
output_directory = args.output_directory
number_additive_traits = args.number_additive_traits
num_epochs_grid = args.num_epochs_grid
num_epochs = args.num_epochs
num_resamplings = args.num_resamplings
early_stopping = args.early_stopping
num_models = args.num_models
random_seed = args.random_seed
#Grid search arguments
l1 = [float(i) for i in args.l1_regularization_factor.split(",")]
l2 = [float(i) for i in args.l2_regularization_factor.split(",")]
batch_size = [int(i) for i in args.num_samples.split(",")]
learn_rate = [float(i) for i in args.learning_rate.split(",")]

#######################################################################
## PACKAGES ##
#######################################################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
#tf.config.optimizer.set_jit(False)
from tensorflow import keras
from keras.models import load_model
import random
from sklearn.metrics import mean_absolute_error
import matplotlib
import os

from keras.constraints import Constraint
from keras.layers import Layer
from keras import backend as K

import jax
import jax.nn as nn
from jax.random import normal
from jax.experimental import sparse
from jax.tree_util import tree_map
import jaxlib
import jax.numpy as jnp
import haiku as hk
import optax
from jax import jit
from functools import partial


#######################################################################
## CLASSES ##
#######################################################################

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

#######################################################################
## FUNCTIONS ##
#######################################################################

def load_model_data(file_dict):
    data_dict = {}
    for name in file_dict.keys():
        # Initialize
        data_dict[name] = {}

        # Read the entire file once
        df = pd.read_csv(file_dict[name])

        # Column names
        ALL_COLUMNS = list(df.columns)
        SELECT_COLUMNS = [col for col in ALL_COLUMNS if col.startswith("dataset_")]
        FOLD_COLUMNS = [col for col in ALL_COLUMNS if col.startswith("fold_") or col == "WT"]
        BIND_COLUMNS = [col for col in ALL_COLUMNS if col.startswith("bind_") or col == "WT"]
        TARGET_COLUMN = "fitness"
        TARGET_SD_COLUMN = "fitness_sd"
        SEQUENCE_COLUMN = "variant_sequence"
        TRAINING_SET_COLUMN = "training_set"

        # Save (sparse) tensors
        data_dict[name]["select"] = jnp.array(df[SELECT_COLUMNS], dtype=jnp.float32)
        data_dict[name]["fold"] = sparse.BCOO.fromdense(jnp.array(df[FOLD_COLUMNS], dtype=jnp.float32))
        data_dict[name]["bind"] = sparse.BCOO.fromdense(jnp.array(df[BIND_COLUMNS], dtype=jnp.float32))
        data_dict[name]["target"] = jnp.array(df[TARGET_COLUMN], dtype=jnp.float32)
        data_dict[name]["target_sd"] = jnp.array(df[TARGET_SD_COLUMN], dtype=jnp.float32)

        # Save remaining columns
        if SEQUENCE_COLUMN in df.columns:
            data_dict[name]["sequence"] = np.array(df[SEQUENCE_COLUMN].values)
        if TRAINING_SET_COLUMN in df.columns:
            data_dict[name]["training_set"] = jnp.expand_dims(jnp.array(df[TRAINING_SET_COLUMN].values), axis=-1)

        data_dict[name]["fold_colnames"] = np.array([col.replace("fold_", "") for col in FOLD_COLUMNS])
        data_dict[name]["bind_colnames"] = np.array([col.replace("bind_", "") for col in BIND_COLUMNS])

    return data_dict

#Resample training data
def resample_training_data_jax(tensor_dict, n_resamplings, rng):
    # Resample observed fitness from error distribution

    observed_fitness = tensor_dict["target"]
    observed_fitness_sd = tensor_dict["target_sd"]

    observed_fitness_resample = jnp.array(
    [jnp.array(
      [observed_fitness[i]+(observed_fitness_sd[i] * jax.random.normal(rng, shape=(1,))) for i in range(len(observed_fitness))])
    for j in range(n_resamplings)]
    )
    print('here')
    #Save new data

    tensor_dict["target"] = jax.device_put(jnp.expand_dims(observed_fitness_resample.ravel(), -1))

    select_tensors = [tensor_dict["select"] for i in range(n_resamplings)]  # Assuming n_resamplings is defined
    tensor_dict["select"] = jnp.concatenate(select_tensors, axis=0)

    fold_matrices = [tensor_dict["fold"] for i in range(n_resamplings)]  # Assuming fold tensors are JAX-compatible sparse matrices
    tensor_dict["fold"] = jax.experimental.sparse.bcoo_concatenate(fold_matrices, dimension=0)

    bind_matrices = [tensor_dict["bind"] for i in range(n_resamplings)]  # Assuming bind tensors are JAX-compatible sparse matrices
    tensor_dict["bind"] = jax.experimental.sparse.bcoo_concatenate(bind_matrices, dimension=0)

    return tensor_dict

#Get sequence ID from sequence string
def get_seq_id(sq):
    return ":".join([str(i)+sq[i] for i in range(len(sq))])

#Little function that returns layer index corresponding to layer name
def get_layer_index(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx

def _shuffle_array(rng, arr):
    flat_arr = arr.ravel()
    shuffled_flat_arr = jax.random.permutation(rng, flat_arr)
    return jnp.reshape(shuffled_flat_arr, arr.shape)

def shuffle_weights(rng, model, weights=None):

    """
    Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
    (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
    If `None`, permute the model's current weights.
    """

    if weights is None:
        weights = model.params

    rngs = jax.random.split(rng, len(weights))
    shuffled_weights = tree_map(lambda r, w: _shuffle_array(r, w), rngs, weights)

    # Assuming `model` has a method `replace` that replaces its parameters.
    # Adjust this line if the model structure is different.
    new_model = model.replace(params=shuffled_weights)
    return new_model

#might remove
def between(min_value, max_value):
    def clip_fn(params):
        return jax.tree_map(lambda w: jnp.clip(w, min_value, max_value), params)
    return clip_fn


def create_model_fn(number_additive_traits, l1, l2):
    """
    This function returns a function that creates the model. The model is a
    neural network that predicts the log fold change of a protein. The model
    uses a combination of linear and nonlinear layers. The nonlinear layers
    use additive traits, which are the outputs of the linear layers.
    The outputs of the nonlinear layers are then used to create a multiplicative
    layer that is used in the prediction.
    """
    def model_fn(inputs_select, inputs_folding, inputs_binding):
        """
        This function creates the model. The model is a neural network that
        predicts the log fold change of a protein. The model uses a combination
        of linear and nonlinear layers. The nonlinear layers use additive traits,
        which are the outputs of the linear layers. The outputs of the nonlinear
        layers are then used to create a multiplicative layer that is used in
        the prediction.
        """
        input_layer_select_folding = jnp.expand_dims(inputs_select[:, 0], -1)
        input_layer_select_binding = jnp.expand_dims(inputs_select[:, 1], -1)

        folding_additive_trait_layer = hk.Linear(number_additive_traits, w_init=hk.initializers.GlorotNormal(), with_bias=False)(inputs_folding)
        folding_nonlinear_layer = StateProbFolded()(folding_additive_trait_layer)
        folding_additive_layer = hk.Linear(1,
                                           w_init=hk.initializers.GlorotNormal(),
                                           with_bias=False,
                                           kernel_regularizer=hk.regularizers.L1L2(l1=l1, l2=l2))(folding_nonlinear_layer)

        binding_additive_trait_layer = hk.Linear(number_additive_traits, w_init=hk.initializers.GlorotNormal(), with_bias=False)(inputs_binding)
        binding_nonlinear_layer = StateProbBound()(binding_additive_trait_layer, folding_additive_trait_layer)
        binding_additive_layer = hk.Linear(1,
                                           w_init=hk.initializers.GlorotNormal(),
                                           with_bias=False)(binding_nonlinear_layer)

        multiplicative_layer_folding = folding_additive_layer * input_layer_select_folding
        multiplicative_layer_binding = binding_additive_layer * input_layer_select_binding
        output_layer = multiplicative_layer_folding + multiplicative_layer_binding

        return output_layer

    return model_fn


def create_model(learn_rate, l1, l2, input_dim_select, input_dim_folding, input_dim_binding, number_additive_traits):
    # Create model
    model_fn = create_model_fn(number_additive_traits, l1, l2)
    model = hk.without_apply_rng(hk.transform(model_fn))

    # Create optimizer
    opt = optax.chain(
        optax.adam(learn_rate),
        optax.constraint(between(0, 1e3), ['folding_additive', 'binding_additive']),
    )

    # Create regularizer
    regularizer = hk.regularizers.L1L2(l1=l1, l2=l2)

    return model, opt, regularizer


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

#Fit model for gridsearch
def fit_model_grid(param_dict, input_data, n_epochs, rng):

    #Summarize results
    print("Grid search using %s" % (param_dict))

    rng_init, rng_batches = jax.random.split(rng)

    #Create model
    model, optimizer, regularizer = create_model(
        learn_rate = param_dict['learning_rate'],
        l1=param_dict['l1_regularization_factor'],
        l2=param_dict['l2_regularization_factor'],
        input_dim_select = input_data['train']['select'].shape[1],
        input_dim_folding = input_data['train']['fold'].shape[1],
        input_dim_binding = input_data['train']['bind'].shape[1],
        number_additive_traits = param_dict['number_additive_traits'])

    def loss_fn(params, inputs_select, inputs_folding, inputs_binding, target):
        output = model.apply(params, inputs_select, inputs_folding, inputs_binding)
        loss = jnp.mean(jnp.abs(output - target))
        reg_loss = regularizer(params)
        total_loss = loss + reg_loss
        return total_loss

    @jax.jit
    def update(params, opt_state, inputs_select, inputs_folding, inputs_binding, target):
        grads = jax.grad(loss_fn)(params, inputs_select, inputs_folding, inputs_binding, target)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    params = model.init(rng, input_data['train']['select'], input_data['train']['fold'], input_data['train']['bind'])
    opt_state = optimizer.init(params)

    for epoch in range(n_epochs):
        for batch_data in generate_batches(input_data['train'], param_dict['num_samples'], rng_batches):
            inputs_select, inputs_folding, inputs_binding, target = batch_data
            params, opt_state = update(params, opt_state, inputs_select, inputs_folding, inputs_binding, target)
        # Validation loss
        val_loss = loss_fn(params, input_data['valid']['select'], input_data['valid']['fold'], input_data['valid']['bind'], input_data['valid']['target'])
    return val_loss.item()

#######################################################################
## SETUP ##
#######################################################################

#print(output_directory)

#Output model directory
model_directory = os.path.join(output_directory, "whole_model")
#Create output model directory
try:
    os.mkdir(model_directory)
except FileExistsError:
    print("Warning: Output model directory already exists.")

#Output plot directory
plot_directory = os.path.join(output_directory, "plots")
#Create output plot directory
try:
    os.mkdir(plot_directory)
except FileExistsError:
    print("Warning: Output plot directory already exists.")

#create the rng
random_seed = 42
rng = jax.random.PRNGKey(random_seed)

#Load model data
model_data = load_model_data({
    "train": data_train_file,
    "valid": data_valid_file,
    "obs": data_obs_file
    })

#Resample training data
if num_resamplings!=0:
    model_data["train"] = resample_training_data_jax(
        tensor_dict = model_data["train"],
        n_resamplings = num_resamplings,
        rand_num_gen = rng
        )

#######################################################################
## TUNE LEARNING RATE, NUMBER OF SAMPLES AND REGULARISATION PARAMS ##
#######################################################################

#Fit model
if len(l1) == 1 and len(l2) == 1 and len(batch_size) == 1 and len(learn_rate) == 1:
    best_params = {
        "num_samples": batch_size[0],
        "learning_rate": learn_rate[0],
        "l1_regularization_factor": l1[0],
        "l2_regularization_factor": l2[0],
        "number_additive_traits": 1
    }
else:
    parameter_grid = [{
        "num_samples": i,
        "learning_rate": j,
        "l1_regularization_factor": k,
        "l2_regularization_factor": l,
        "number_additive_traits": 1
    } for i in batch_size for j in learn_rate for k in l1 for l in l2]

    rng = jax.random.PRNGKey(random_seed)
    rngs = jax.random.split(rng, len(parameter_grid))

    grid_results = [fit_model_grid(params, model_data, num_epochs_grid, rng_key) for params, rng_key in zip(parameter_grid, rngs)]

    best_params = parameter_grid[np.argmin(grid_results)]

    print("Best: %f using %s" % (min(grid_results), best_params))

num_samples = best_params['num_samples']
learning_rate = best_params['learning_rate']
l1_regularization_factor = best_params['l1_regularization_factor']
l2_regularization_factor = best_params['l2_regularization_factor']

#######################################################################
## BUILD FINAL NEURAL NETWORK ##
#######################################################################

#create the rng
random_seed = 42
rng = jax.random.PRNGKey(random_seed)

#Create model
model, optimizer, regularizer = create_model(
    learn_rate = learning_rate,
    l1=l1_regularization_factor,
    l2=l2_regularization_factor,
    input_dim_select = model_data['train']['select'].shape[1],
    input_dim_folding = model_data['train']['fold'].shape[1],
    input_dim_binding = model_data['train']['bind'].shape[1],
    number_additive_traits = number_additive_traits)
print(model.summary())

#Validation data
validation_data = (
  [model_data['valid']['select'], model_data['valid']['fold'], model_data['valid']['bind']],
  model_data['valid']['target'])

#Fit model(s)
for model_count in range(num_models):

  #Shuffle model weights
  shuffle_weights(model, original_model_weights)

  #Callbacks
  model_callbacks = []
  if early_stopping:
    model_callbacks = [
      keras.callbacks.EarlyStopping(monitor='val_loss', patience=num_epochs*0.1),
      keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_directory, 'my_model_'+str(model_count)), monitor='val_loss', save_best_only=True)]

  #Fit the model
  history = model.fit(
    [model_data['train']['select'], model_data['train']['fold'], model_data['train']['bind']],
    model_data['train']['target'],
    validation_data = validation_data,
    epochs = num_epochs,
    batch_size = num_samples,
    shuffle = True,
    callbacks = model_callbacks,
    verbose = 2,
    use_multiprocessing = True)

  #Save the entire model as a SavedModel
  if early_stopping==False:
    model.save(os.path.join(model_directory, 'my_model_'+str(model_count)))

  #Load model
  custom = {'Between':Between}
  model = load_model(os.path.join(model_directory, 'my_model_'+str(model_count)))

  #Plot model performance per epoch
  my_figure = plt.figure(figsize = (8,8))
  plt.plot(
    np.log(history.history['loss']))
  plt.xlabel('Number of epochs')
  plt.ylabel('Mean Absolute Error (MAE) on testing data')
  my_figure.savefig(os.path.join(plot_directory, "model_performance_perepoch_"+str(model_count)+".pdf"), bbox_inches='tight')

  #######################################################################
  ## SAVE OBSERVATIONS, PREDICTIONS & ADDITIVE TRAIT VALUES ##
  #######################################################################

  #Model predictions on observed variants
  model_predictions = model.predict([model_data['obs']['select'], model_data['obs']['fold'], model_data['obs']['bind']])

  #Index for folding additive trait layer
  layer_idx_folding = get_layer_index(
    model = model,
    layername = "folding_additivetrait")
  #Calculate folding additive trait
  folding_additive_traits_model = keras.Model(
    inputs = model.input,
    outputs = model.layers[layer_idx_folding].output)
  #Convert to data frame
  folding_additive_trait_df = pd.DataFrame(folding_additive_traits_model.predict([model_data['obs']['select'], model_data['obs']['fold'], model_data['obs']['bind']]))
  folding_additive_trait_df.columns = [ "trait " + str(i) for i in range(len(folding_additive_trait_df.columns))]

  #Index for binding additive trait layer
  layer_idx_binding = get_layer_index(
    model = model,
    layername = "binding_additivetrait")
  #Calculate binding additive trait
  binding_additive_traits_model = keras.Model(
    inputs = model.input,
    outputs = model.layers[layer_idx_binding].output)
  #Convert to data frame
  binding_additive_trait_df = pd.DataFrame(binding_additive_traits_model.predict([model_data['obs']['select'], model_data['obs']['fold'], model_data['obs']['bind']]))
  binding_additive_trait_df.columns = [ "trait " + str(i) for i in range(len(binding_additive_trait_df.columns))]

  #Results data frame
  dataframe_to_export = pd.DataFrame({
    "seq" : np.array(model_data['obs']['sequence']).flatten(),
    "observed_fitness" : np.array(model_data['obs']['target']).flatten(),
    "predicted_fitness" : model_predictions.flatten(),
    "additive_trait_folding" : folding_additive_trait_df["trait 0"],
    "additive_trait_binding" : binding_additive_trait_df["trait 0"],
    "training_set" : np.array(model_data['obs']['training_set']).flatten()})
  #Save as csv file
  dataframe_to_export.to_csv(
    os.path.join(output_directory, "predicted_fitness_"+str(model_count)+".txt"),
    sep = "\t",
    index = False)

  #Save model weights
  dataframe_to_export_folding = pd.DataFrame({
    "id" : model_data['obs']['fold_colnames'],
    "folding_coefficient" : [i[0] for i in model.layers[layer_idx_folding].get_weights()[0]]})
  dataframe_to_export_binding = pd.DataFrame({
    "id" : model_data['obs']['bind_colnames'],
    "binding_coefficient" : [i[0] for i in model.layers[layer_idx_binding].get_weights()[0]]})
  #Merge
  dataframe_to_export = dataframe_to_export_folding.merge(dataframe_to_export_binding, left_on='id', right_on='id', how='outer')
  #Save as csv file
  dataframe_to_export.to_csv(
    os.path.join(output_directory, "model_weights_"+str(model_count)+".txt"),
    sep = "\t",
    index = False)

  #Save remaining model parameters (linear layers)
  with open(os.path.join(output_directory, "model_parameters_"+str(model_count)+".txt"), 'w') as f:
    for ml in model.layers:
      if(ml.name in ["folding_additive", "binding_additive"]):
        f.write(ml.name.replace("additive", "linear")+"_kernel\n")
        f.write(str(float(ml.weights[0]))+"\n")
        f.write(ml.name.replace("additive", "linear")+"_bias\n")
        f.write(str(float(ml.weights[1]))+"\n")
