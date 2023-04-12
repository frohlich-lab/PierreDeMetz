
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
from tensorflow import keras
from keras.models import load_model
import random
from sklearn.metrics import mean_absolute_error
import matplotlib
import os
from keras import layers, Model, callbacks

from keras.constraints import Constraint
from keras.layers import Layer
from keras import backend as K

from keras.callbacks import LambdaCallback
from keras.models import load_model


#Generate dataset
def load_model_data(file_dict):
  data_dict = {}
  for name in file_dict.keys():
    #Initialise
    data_dict[name] = {}
    #Column names
    ALL_COLUMNS = list(pd.read_csv(file_dict[name], nrows = 1).columns)
    SELECT_COLUMNS = [i for i in range(len(ALL_COLUMNS)) if str.startswith(ALL_COLUMNS[i], "dataset_")]
    FOLD_COLUMNS = [i for i in range(len(ALL_COLUMNS)) if str.startswith(ALL_COLUMNS[i], "fold_") or ALL_COLUMNS[i]=="WT"]
    BIND_COLUMNS = [i for i in range(len(ALL_COLUMNS)) if str.startswith(ALL_COLUMNS[i], "bind_") or ALL_COLUMNS[i]=="WT"]
    TARGET_COLUMN = [i for i in range(len(ALL_COLUMNS)) if ALL_COLUMNS[i]=="fitness"]
    TARGET_SD_COLUMN = [i for i in range(len(ALL_COLUMNS)) if ALL_COLUMNS[i]=="fitness_sd"]
    SEQUENCE_COLUMN = [i for i in range(len(ALL_COLUMNS)) if ALL_COLUMNS[i]=="variant_sequence"]
    TRAINING_SET_COLUMN = [i for i in range(len(ALL_COLUMNS)) if ALL_COLUMNS[i]=="training_set"]
    #Save (sparse) tensors
    data_dict[name]["select"] = tf.convert_to_tensor(np.asarray(pd.read_csv(file_dict[name], usecols = SELECT_COLUMNS)), np.float32)
    data_dict[name]["fold"] = tf.sparse.from_dense(tf.convert_to_tensor(np.asarray(pd.read_csv(file_dict[name], usecols = FOLD_COLUMNS)), np.float32))
    data_dict[name]["bind"] = tf.sparse.from_dense(tf.convert_to_tensor(np.asarray(pd.read_csv(file_dict[name], usecols = BIND_COLUMNS)), np.float32))
    data_dict[name]["target"] = tf.convert_to_tensor(np.asarray(pd.read_csv(file_dict[name], usecols = TARGET_COLUMN)), np.float32)
    data_dict[name]["target_sd"] = tf.convert_to_tensor(np.asarray(pd.read_csv(file_dict[name], usecols = TARGET_SD_COLUMN)), np.float32)
    #Save remaining columns
    if len(SEQUENCE_COLUMN)!=0 and len(TRAINING_SET_COLUMN)!=0:
      data_dict[name]["sequence"] = np.asarray(pd.read_csv(file_dict[name], usecols = SEQUENCE_COLUMN))
    if len(TRAINING_SET_COLUMN)!=0:
      data_dict[name]["training_set"] = np.asarray(pd.read_csv(file_dict[name], usecols = TRAINING_SET_COLUMN))
    data_dict[name]["fold_colnames"] = np.asarray([ALL_COLUMNS[i].replace("fold_", "") for i in FOLD_COLUMNS])
    data_dict[name]["bind_colnames"] = np.asarray([ALL_COLUMNS[i].replace("bind_", "") for i in BIND_COLUMNS])
  return data_dict

#Load our model
loaded_model = load_model('/Users/pierredemetz/UCL_work/Crick/doubledeepms/Results/Data/mochi/GRB2-SH3/mochi__fit_tmodel_3state_sparse_dimsum128_subsample50p/whole_model/my_model_0/')


#Load model data
model_data = load_model_data({
  "train": data_train_file,
  "valid": data_valid_file,
  "obs": data_obs_file})



#Make prediction
folding_additive_model = keras.Model(
    inputs = loaded_model.input,
    outputs = loaded_model.layers[7].output)

binding_additive_model = keras.Model(
    inputs = loaded_model.input,
    outputs = loaded_model.layers[9].output)

output_folding = folding_additive_model.predict([model_data['obs']['select'], model_data['obs']['fold'], model_data['obs']['bind']],
                                                batch_size = 1,
                                                #shuffle = False
                                                )
#print('output folding')
#print(output_folding)
#print(type(output_folding))
#print(' ')
output_binding = binding_additive_model.predict([model_data['obs']['select'], model_data['obs']['fold'], model_data['obs']['bind']],
                                                batch_size=1,
                                                #shuffle = False
                                                )
#print('output binding')
#print(output_binding)


# Combine the two arrays horizontally
combined_array = np.hstack((output_folding, output_binding))

# Create a pandas DataFrame from the combined array
df = pd.DataFrame(combined_array)

# Set the column names
column_names = ['folding_additive_layer_output', 'binding_additive_layer_output']
df.columns = column_names

# Save the DataFrame to a CSV file
df.to_csv("arrays_keras.csv", index=False)
