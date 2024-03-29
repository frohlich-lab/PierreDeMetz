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
parser.add_argument("--model_type", "-mt", default = 'tri_state_explicit', help = "Model type (default:tri_state_explicit)")
parser.add_argument("--union_mode", "-um", default = 'False', help = "Union mode (default:union)")
parser.add_argument("--protein", '-prot', default = 'GRB2', help = "Protein name (default:GRB2)")
parser.add_argument("--wandb", '-w', default = 'False', help = "Whether to use wandb (default:False)")
parser.add_argument("--project_name", '-pn', default = 'pierre_mochi__fit_tmodel_3state_doubledeepms', help = "Wandb project name (default:pierre_mochi__fit_tmodel_3state_doubledeepms)")
parser.add_argument('--is_implicit', action='store_true', default=False, help='Set is_implicit as True')
parser.add_argument('--is_complex', action='store_true', default=False, help='Set is_degradation as True')
#parser.add_argument('--is_degradation', action='store_true', default=False, help='Set is_degradation as True')


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
model_type = args.model_type
union_mode = args.union_mode
protein = args.protein
wandb_status = args.wandb
project_name = args.project_name
is_implicit = args.is_implicit
is_complex = args.is_complex
#is_degradation = args.is_degradation
#specs = (is_implicit, is_degradation)

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
import os

import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial

from utils import constrained_gradients, StateProbBound, StateProbFolded, Between, apply_weight_constraints, shuffle_weights
from training import model_training, fit_model_grid_jax, fit_model_grid_complex, model_training_complex
from dataloading import load_model_data_jax, resample_training_data_jax, load_model_data_complex
from model_creation import create_model_fn, create_model_jax

#######################################################################
## SETUP ##
#######################################################################

import os
os.environ["WANDB__SERVICE_WAIT"] = "3600"

#Output model directory
model_directory = os.path.join(output_directory, "whole_model")
#Create output model directory
try:
    os.mkdir(model_directory,)
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


wandb_config = {
    'number_additive_traits': args.number_additive_traits,
    'num_epochs_grid': args.num_epochs_grid,
    'num_epochs': args.num_epochs,
    'num_resamplings': args.num_resamplings,
    'num_models': args.num_models,
    'random_seed': args.random_seed,
    'model_type': args.model_type,
    'union_mode': args.union_mode,
    'protein': args.protein,
    'project_name': args.project_name,
    'status': args.wandb
}


rngs = hk.PRNGSequence(jax.random.PRNGKey(42))

if is_complex==True:
    model_data_jax = load_model_data_complex({
        "train": data_train_file,
        "valid": data_valid_file,
        "obs": data_obs_file
        },
                                        union_mode
                                        )
else:
    model_data_jax = load_model_data_jax({
        "train": data_train_file,
        "valid": data_valid_file,
        "obs": data_obs_file
        },
                                        union_mode
                                        )

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
        "number_additive_traits": 1,
        "model_type": model_type,
        "is_implicit": is_implicit,
        "is_complex": is_complex
    }

else:
    parameter_grid = [{
        "num_samples": i,
        "learning_rate": j,
        "l1_regularization_factor": k,
        "l2_regularization_factor": l,
        "number_additive_traits": 1,
        "model_type": model_type,
        "is_implicit": is_implicit,
        "is_complex": is_complex
    } for i in batch_size for j in learn_rate for k in l1 for l in l2]

    rng = jax.random.PRNGKey(random_seed)
    rngs_grid = jax.random.split(rng, len(parameter_grid))

if is_complex==True:
    grid_results = [
        fit_model_grid_complex(
            params,
            model_data_jax,
            num_epochs_grid,
            rng_key,
            {**wandb_config, 'run_number': i+1}
        )
        for i, (params, rng_key) in enumerate(zip(parameter_grid, rngs_grid))
    ]

    best_params = parameter_grid[np.argmin(grid_results)]

    print("Best: %f using %s" % (min(grid_results), best_params))

elif is_complex==False:
    grid_results = [
        fit_model_grid_jax(
            params,
            model_data_jax,
            num_epochs_grid,
            rng_key,
            {**wandb_config, 'run_number': i+1}
        )
        for i, (params, rng_key) in enumerate(zip(parameter_grid, rngs_grid))
    ]

    best_params = parameter_grid[np.argmin(grid_results)]

    print("Best: %f using %s" % (min(grid_results), best_params))

#######################################################################
## BUILD FINAL NEURAL NETWORK ##
#######################################################################


model, opt_init, opt_update = create_model_jax(
    rng=rng,
    learn_rate=best_params['learning_rate'],
    l1=best_params['l1_regularization_factor'],
    l2=best_params['l2_regularization_factor'],
    number_additive_traits=best_params['number_additive_traits'],
    model_type=best_params['model_type'],
    is_implicit=best_params['is_implicit'],
    is_complex=best_params['is_complex']
)

if is_complex==True:
    weights = model.init(rng,
                        jnp.ones_like(model_data_jax['train']['select']),
                        jnp.ones_like(model_data_jax['train']['fold_mutation']),
                        jnp.ones_like(model_data_jax['train']['fold_location']),
                        jnp.ones_like(model_data_jax['train']['bind_mutation']),
                        jnp.ones_like(model_data_jax['train']['bind_location']))
elif is_complex==False:
    weights = model.init(next(rngs),
                    jnp.ones_like(model_data_jax['train']['select']),
                    jnp.ones_like(model_data_jax['train']['fold']),
                    jnp.ones_like(model_data_jax['train']['bind']))

opt_state = opt_init(weights)

weights = apply_weight_constraints(weights, 'folding_additive', 0, 1e3)
weights = apply_weight_constraints(weights, 'binding_additive', 0, 1e3)
if is_complex==True:
    history, model, trained_weights = model_training_complex(model,
                                                    opt_state,
                                                    opt_update,
                                                    weights, best_params, model_data_jax, num_epochs, next(rngs),
                                                    wandb_config)
if is_complex==False:
    history, model, trained_weights = model_training(model,
                                                    opt_state,
                                                    opt_update,
                                                    weights, best_params, model_data_jax, num_epochs, next(rngs),
                                                    wandb_config)

if is_complex==True:
    model_outputs= model.apply(trained_weights,
                                    model_data_jax['obs']['select'],
                                    model_data_jax['obs']['fold_mutation'],
                                    model_data_jax['obs']['fold_location'],
                                    model_data_jax['obs']['bind_mutation'],
                                    model_data_jax['obs']['bind_location']
                                    )
elif is_complex==False:
    model_outputs= model.apply(trained_weights,
                               model_data_jax['obs']['select'],
                               model_data_jax['obs']['fold'],
                               model_data_jax['obs']['bind'])

prediction,folding_additive_layer_output, binding_additive_layer_output, folding_additive_trait_layer_outputs, binding_additive_trait_layer_outputs = model_outputs

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
if is_complex==True:
    model_outputs= model.apply(trained_weights,
                                    model_data_jax['obs']['select'],
                                    model_data_jax['obs']['fold_mutation'],
                                    model_data_jax['obs']['fold_location'],
                                    model_data_jax['obs']['bind_mutation'],
                                    model_data_jax['obs']['bind_location']
                                    )
elif is_complex==False:
    model_outputs= model.apply(trained_weights,
                               model_data_jax['obs']['select'],
                               model_data_jax['obs']['fold'],
                               model_data_jax['obs']['bind'])

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
