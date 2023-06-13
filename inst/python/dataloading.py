import numpy as np
import jax
import pandas as pd
import jax.numpy as jnp
import pprint

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

def load_model_data_complex(file_dict, union_mode=False):
    amino_acid_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    data_dict = {}
    for name in file_dict.keys():
        data_dict[name] = {}
        df = pd.read_csv(file_dict[name])

        ALL_COLUMNS = list(df.columns)
        SELECT_COLUMNS = [col for col in ALL_COLUMNS if col.startswith("dataset_")]
        FOLD_COLUMNS = [col for col in ALL_COLUMNS if col.startswith("fold_") or col == "WT"]
        BIND_COLUMNS = [col for col in ALL_COLUMNS if col.startswith("bind_") or col == "WT"]
        TARGET_COLUMN = "fitness"
        TARGET_SD_COLUMN = "fitness_sd"
        SEQUENCE_COLUMN = "variant_sequence"
        TRAINING_SET_COLUMN = "training_set"

        mutation_matrix_bind = jnp.zeros((len(df), 20, 20), dtype=jnp.float32)
        mutation_matrix_fold = jnp.zeros((len(df), 20, 20), dtype=jnp.float32)
        location_matrix_bind = jnp.zeros((len(df), len(df.columns) - len(BIND_COLUMNS)), dtype=jnp.float32)
        location_matrix_fold = jnp.zeros((len(df), len(df.columns) - len(FOLD_COLUMNS)), dtype=jnp.float32)

        import re

        for i, col in enumerate(BIND_COLUMNS):
            if col == "WT":
                continue
            stripped_col = col.replace("bind_", "")
            match = re.match(r"([A-Z])(\d+)([A-Z])", stripped_col)
            if match:
                original, position, mutation = match.groups()
                mutation_matrix_bind = mutation_matrix_bind.at[:, amino_acid_to_index[original], amino_acid_to_index[mutation]].set(df[col])
                location_matrix_bind = location_matrix_bind.at[:, int(position)-1].set(df[col])
            else:
                print(f"Could not parse column name: {col}")

        for i, col in enumerate(FOLD_COLUMNS):
            if col == "WT":
                continue
            stripped_col = col.replace("fold_", "")
            match = re.match(r"([A-Z])(\d+)([A-Z])", stripped_col)
            if match:
                original, position, mutation = match.groups()
                mutation_matrix_fold = mutation_matrix_fold.at[:, amino_acid_to_index[original], amino_acid_to_index[mutation]].set(df[col])
                location_matrix_fold = location_matrix_fold.at[:, int(position)-1].set(df[col])
            else:
                print(f"Could not parse column name: {col}")

        data_dict[name]["select"] = jnp.array(df[SELECT_COLUMNS], dtype=jnp.float32)
        data_dict[name]["bind_mutation"] = mutation_matrix_bind
        data_dict[name]["bind_location"] = location_matrix_bind
        data_dict[name]["fold_mutation"] = mutation_matrix_fold
        data_dict[name]["fold_location"] = location_matrix_fold
        data_dict[name]["target"] = jnp.array(df[TARGET_COLUMN], dtype=jnp.float32)
        data_dict[name]["target_sd"] = jnp.array(df[TARGET_SD_COLUMN], dtype=jnp.float32)

        if SEQUENCE_COLUMN in df.columns:
            data_dict[name]["sequence"] = np.array(df[SEQUENCE_COLUMN].values)
        if TRAINING_SET_COLUMN in df.columns:
            data_dict[name]["training_set"] = jnp.expand_dims(jnp.array(df[TRAINING_SET_COLUMN].values), axis=-1)

        data_dict[name]["fold_colnames"] = np.array([col.replace("fold_", "") for col in FOLD_COLUMNS])
        data_dict[name]["bind_colnames"] = np.array([col.replace("bind_", "") for col in BIND_COLUMNS])

        if union_mode == 'True':
            data_dict[name] = create_union_dataset(data_dict[name])

    return data_dict

if __name__ == '__main__':
    union_mode = False

    data_train_file = '/Users/pierredemetz/UCL_work/Crick/doubledeepms/Results//Data/mochi/GRB2-SH3/mochi__fit_tmodel_3state_sparse_dimsum128_subsample100p//dataset_train.txt'
    data_valid_file = '/Users/pierredemetz/UCL_work/Crick/doubledeepms/Results//Data/mochi/GRB2-SH3/mochi__fit_tmodel_3state_sparse_dimsum128_subsample100p//dataset_valid.txt'
    data_obs_file = '/Users/pierredemetz/UCL_work/Crick/doubledeepms/Results//Data/mochi/GRB2-SH3/mochi__fit_tmodel_3state_sparse_dimsum128_subsample100p//dataset_all.txt'

    pprint.pprint(load_model_data_complex({
    "train": data_train_file,
    "valid": data_valid_file,
    "obs": data_obs_file
    },
                                     union_mode
                                     ))
