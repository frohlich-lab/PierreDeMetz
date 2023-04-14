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
from tensorflow import keras
from keras.models import load_model
import os

import jax
from jax.experimental import sparse
from jax.tree_util import tree_map
import jax.numpy as jnp
import haiku as hk
import optax
from jax import jit
import pickle
from functools import partial

from utils import constrained_gradients, StateProbBound, StateProbFolded, Between, apply_weight_constraints, shuffle_weights
from training import model_training, fit_model_grid_jax
from dataloading import load_model_data_jax, resample_training_data_jax
from model_creation import create_model_fn, create_model_jax
from weights_loading import weights_loading
#######################################################################
## SETUP ##
#######################################################################


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

#Weights directory
weights_directory = os.path.join(output_directory, "weights")
#Create output plot directory
try:
  os.mkdir(weights_directory)
except FileExistsError:
  print("Warning: Output weights directory already exists.")

#Boostrap directory
bootstrap_directory = os.path.join(output_directory, "bootstrap")
#Create output plot directory
try:
  os.mkdir(bootstrap_directory)
except FileExistsError:
  print("Warning: Output boostrap directory already exists.")


rngs = hk.PRNGSequence(jax.random.PRNGKey(42))


#Load model data
model_data_jax = load_model_data_jax({
    "train": data_train_file,
    "valid": data_valid_file,
    "obs": data_obs_file
    })

#Resample training data
if num_resamplings!=0:
    model_data_jax["train"] = resample_training_data_jax(
        tensor_dict = model_data_jax["train"],
        n_resamplings = num_resamplings,
        rng = next(rngs)
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
    #print(len(parameter_grid))
    grid_results = [fit_model_grid_jax(params, model_data_jax, num_epochs_grid, rng_key) for params, rng_key in zip(parameter_grid, rngs)]
    #grid_results = [fit_model_grid_jax(params, model_data_jax, num_epochs_grid, rng) for params in parameter_grid]

    best_params = parameter_grid[np.argmin(grid_results)]

    print("Best: %f using %s" % (min(grid_results), best_params))


#######################################################################
## BUILD FINAL NEURAL NETWORK ##
#######################################################################


model, opt_init, opt_update = create_model_jax(
    rng=next(rngs),
    learn_rate=best_params['learning_rate'],
    l1=best_params['l1_regularization_factor'],
    l2=best_params['l2_regularization_factor'],
    input_dim_select=model_data_jax['train']['select'].shape[1],
    input_dim_folding=model_data_jax['train']['fold'].shape[1],
    input_dim_binding=model_data_jax['train']['bind'].shape[1],
    number_additive_traits=best_params['number_additive_traits']
)

#weights = model.init(next(rngs), model_data_jax['train']['select'], model_data_jax['train']['fold'], model_data_jax['train']['bind'])
weights = model.init(next(rngs),
                     jnp.ones_like(model_data_jax['train']['select']),
                     jnp.ones_like(model_data_jax['train']['fold']),
                     jnp.ones_like(model_data_jax['train']['bind']))

opt_state = opt_init(weights)

weights = apply_weight_constraints(weights, 'folding_additive', 0, 1e3)
weights = apply_weight_constraints(weights, 'binding_additive', 0, 1e3)

history, model, trained_weights = model_training(model,
                                                 opt_state,
                                                 opt_update,
                                                 weights, best_params, model_data_jax, num_epochs, next(rngs))

# Model predictions on observed variants
model_outputs= model.apply(trained_weights,
                                model_data_jax['obs']['select'],
                                model_data_jax['obs']['fold'],
                                model_data_jax['obs']['bind']
                                )

prediction ,folding_additive_layer_output, binding_additive_layer_output, folding_additive_trait_layer_outputs, binding_additive_trait_layer_outputs = model_outputs


model_count = 1

#Plot model performance per epoch
my_figure = plt.figure(figsize = (8,8))
plt.plot(np.log(history))
plt.xlabel('Number of epochs')
plt.ylabel('Mean Absolute Error (MAE) on testing data')
my_figure.savefig(os.path.join(plot_directory, "model_performance_perepoch_"+str(model_count)+".pdf"), bbox_inches='tight')

#######################################################################
## SAVE OBSERVATIONS, PREDICTIONS & ADDITIVE TRAIT VALUES ##
#######################################################################

# Model predictions on observed variants
model_outputs = model.apply(trained_weights,
                            model_data_jax['obs']['select'],
                            model_data_jax['obs']['fold'],
                            model_data_jax['obs']['bind']
                        )

prediction,folding_additive_layer_output, binding_additive_layer_output, folding_additive_trait_layer_outputs, binding_additive_trait_layer_outputs = model_outputs

folding_additive_trait_df = pd.DataFrame(folding_additive_trait_layer_outputs)
binding_additive_trait_df = pd.DataFrame(binding_additive_trait_layer_outputs)

#Results data frame
dataframe_to_export = pd.DataFrame({
    "seq" : np.array(model_data_jax['obs']['sequence']).flatten(),
    "observed_fitness" : np.array(model_data_jax['obs']['target']).flatten(),
    "predicted_fitness" : prediction.flatten(),
    "additive_trait_folding" : folding_additive_trait_df[0],
    "additive_trait_binding" : binding_additive_trait_df[0],
    "training_set" : np.array(model_data_jax['obs']['training_set']).flatten()
})

#Save as csv file
dataframe_to_export.to_csv(os.path.join(output_directory, "predicted_fitness_"+str(model_count)+".txt"),
                        sep = "\t",
                        index = False)


# Save model weights
dataframe_to_export_folding = pd.DataFrame({
    "id" : model_data_jax['obs']['fold_colnames'],
    "folding_coefficient" : [jnp.squeeze(trained_weights['folding_additive_trait']['w'][_]) for _ in range(len(model_data_jax['obs']['fold_colnames']))]})
dataframe_to_export_binding = pd.DataFrame({
    "id" : model_data_jax['obs']['bind_colnames'],
    "binding_coefficient" : [jnp.squeeze(trained_weights['binding_additive_trait']['w'][_]) for _ in range(len(model_data_jax['obs']['bind_colnames']))]})

# Save dataframes as csv files
#Merge
dataframe_to_export = dataframe_to_export_folding.merge(dataframe_to_export_binding, left_on='id', right_on='id', how='outer')
#Save as csv file
dataframe_to_export.to_csv(os.path.join(output_directory, "model_weights_"+str(model_count)+".txt"),
                        sep = "\t",
                        index = False)

# Save remaining model parameters (linear layers)
with open(os.path.join(output_directory, "model_parameters_"+str(model_count)+".txt"), 'w') as f:
    for module_name, module_params in trained_weights.items():
        if module_name in ["folding_additive", "binding_additive"]:
            f.write(module_name.replace("additive", "linear")+"_kernel\n")
            f.write(str(float(module_params["w"]))+"\n")
            f.write(module_name.replace("additive", "linear")+"_bias\n")
            f.write(str(float(module_params["b"]))+"\n")
