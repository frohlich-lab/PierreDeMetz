import pandas as pd
import numpy
import os
import jax.numpy as jnp

def weights_loading(output_directory, model_count):

    weights_filepath = os.path.join(output_directory, "model_weights_"+str(model_count)+".txt")
    weights_dataframe = pd.read_csv(weights_filepath, sep="\t")


    # Extract the folding and binding coefficients
    folding_coefficients = weights_dataframe['folding_coefficient'].to_numpy()
    binding_coefficients = weights_dataframe['binding_coefficient'].dropna().to_numpy()

    # Load the remaining model parameters (linear layers) from the separate text file
    parameters_filepath = os.path.join(output_directory, "model_parameters_"+str(model_count)+".txt")

    loaded_linear_params = {}
    with open(parameters_filepath, 'r') as f:
        while True:
            module_name = f.readline().strip()
            if not module_name:
                break
            w = float(f.readline().strip())
            f.readline()  # Skip the next line containing the bias name
            b = float(f.readline().strip())
            loaded_linear_params[module_name] = {'w': jnp.array([[w]]), 'b': jnp.array([b])}

    # Create a dictionary with the loaded weights
    loaded_weights = {
        'folding_additive_trait': {
            'w': folding_coefficients.reshape(-1, 1)
        },
        'folding_additive': loaded_linear_params['folding_linear_kernel'],
        'binding_additive_trait': {
            'w': binding_coefficients.reshape(-1, 1)
        },
        'binding_additive': loaded_linear_params['binding_linear_kernel']
    }
    return loaded_weights
