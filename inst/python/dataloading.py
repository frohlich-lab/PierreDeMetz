import numpy as np
import jax
import pandas as pd
import jax.numpy as jnp

def load_model_data_jax(file_dict, union_mode=False):
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
        data_dict[name]["fold"] = jnp.array(df[FOLD_COLUMNS], dtype=jnp.float32)
        data_dict[name]["bind"] = jnp.array(df[BIND_COLUMNS], dtype=jnp.float32)
        data_dict[name]["target"] = jnp.array(df[TARGET_COLUMN], dtype=jnp.float32)
        data_dict[name]["target_sd"] = jnp.array(df[TARGET_SD_COLUMN], dtype=jnp.float32)
        #print(data_dict)

        # Save remaining columns
        if SEQUENCE_COLUMN in df.columns:
            data_dict[name]["sequence"] = np.array(df[SEQUENCE_COLUMN].values)
        if TRAINING_SET_COLUMN in df.columns:
            data_dict[name]["training_set"] = jnp.expand_dims(jnp.array(df[TRAINING_SET_COLUMN].values), axis=-1)

        data_dict[name]["fold_colnames"] = np.array([col.replace("fold_", "") for col in FOLD_COLUMNS])
        data_dict[name]["bind_colnames"] = np.array([col.replace("bind_", "") for col in BIND_COLUMNS])

        if union_mode == 'True':
            data_dict[name] = create_union_dataset(data_dict[name])

    return data_dict

def create_union_dataset(data_dict):

    # Assuming 'data' is your dictionary
    fold_colnames = data_dict['fold_colnames']
    bind_colnames = data_dict['bind_colnames']

    # Create a set for quick lookup
    fold_colnames_set = set(fold_colnames)
    bind_colnames_set = set(bind_colnames)

    # Find the columns that are in fold_colnames but not in bind_colnames
    new_columns_bind = [col for col in fold_colnames if col not in bind_colnames_set]
    bind = data_dict['bind']

    # Add new columns to 'bind'
    for col in new_columns_bind:
        bind = np.column_stack((bind, np.zeros(bind.shape[0])))

    data_dict['bind'] = bind

    # Now, 'bind' has the same number of columns as 'fold', and 'fold_colnames' can be used for both 'fold' and 'bind'
    data_dict['bind_colnames'] = fold_colnames

    return data_dict

#Resample training data
def resample_training_data_jax(tensor_dict, n_resamplings, rng):
    # Resample observed fitness from error distribution

    observed_fitness = tensor_dict["target"]
    observed_fitness_sd = tensor_dict["target_sd"]

    rngs = jax.random.split(rng, len(observed_fitness_sd))

    observed_fitness_resample = jnp.array(
    [jnp.array(
      [observed_fitness[i]+(observed_fitness_sd[i] * jax.random.normal(rngs[i], shape=(1,))) for i in range(len(observed_fitness))])
    for j in range(n_resamplings)]
    )
    #Save new data

    tensor_dict["target"] = jax.device_put(observed_fitness_resample.ravel())

    #tensor_dict["target"] = jax.device_put(jnp.squeeze(jnp.expand_dims(observed_fitness_resample.ravel(), -1)))

    select_tensors = [tensor_dict["select"] for i in range(n_resamplings)]  # Assuming n_resamplings is defined
    tensor_dict["select"] = jnp.concatenate(select_tensors, axis=0)

    fold_matrices = [tensor_dict["fold"] for i in range(n_resamplings)]
    tensor_dict["fold"] = jnp.concatenate(fold_matrices, axis=0)

    bind_matrices = [tensor_dict["bind"] for i in range(n_resamplings)]
    tensor_dict["bind"] = jnp.concatenate(bind_matrices, axis=0)

    return tensor_dict
