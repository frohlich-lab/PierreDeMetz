import numpy as np
import jax
import pandas as pd
import jax.numpy as jnp
import pprint
import re
import pickle


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

def process_row(row, amino_acid_to_index, column_set, prefix):
    for col in column_set:
        if row[col] == 1:
            stripped_col = col.replace(prefix, "")
            match = re.match(r"([A-Z])(\d+)([A-Z])", stripped_col)
            if match:
                original, position, mutation = match.groups()
                result = pd.Series({
                    'mutation_matrix_index': (row.name, amino_acid_to_index[original], amino_acid_to_index[mutation]),
                    'location_matrix_index': (row.name, int(position)-1),
                })
                return result
    return pd.Series({'mutation_matrix_index': None, 'location_matrix_index': None})


def load_model_data_complex(file_dict, union_mode=False):
    amino_acid_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    data_dict = {}

    for name in file_dict.keys():
        #print(name)
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

        position_values = []
        for col in (BIND_COLUMNS + FOLD_COLUMNS):
            if col == "WT":
                continue
            stripped_col = col.replace("bind_", "").replace("fold_", "")
            match = re.match(r"([A-Z])(\d+)([A-Z])", stripped_col)
            if match:
                original, position, mutation = match.groups()
                position_values.append(int(position))

        num_residues = max(position_values)

        mutation_matrix_bind = np.zeros((len(df), 20, 20), dtype=jnp.float32)
        mutation_matrix_fold = np.zeros((len(df), 20, 20), dtype=jnp.float32)
        location_matrix_bind = np.zeros((len(df), num_residues), dtype=jnp.float32)
        location_matrix_fold = np.zeros((len(df), num_residues), dtype=jnp.float32)

        # create matrix indices using df.apply()
        df_result = df.apply(lambda row: process_row(row, amino_acid_to_index, BIND_COLUMNS, "bind_"), axis=1)
        print('one done')
        #print(df_result)
        # Set 1 at the mutation position in the mutation and location matrices


        for idx, row in df_result.iterrows():

            #print(row['mutation_matrix_index'])
            if row['mutation_matrix_index'] is not None:
                mutation_matrix_bind[row['mutation_matrix_index']]=1
                #print(mutation_matrix_bind[row['mutation_matrix_index']])
                location_matrix_bind[row['location_matrix_index']]=1
                #print(row)

        # create matrix indices using df.apply()
        df_result = df.apply(lambda row: process_row(row, amino_acid_to_index, FOLD_COLUMNS, "fold_"), axis=1)

        # Set 1 at the mutation position in the mutation and location matrices
        for idx, row in df_result.iterrows():

            if row['mutation_matrix_index'] is not None:
                mutation_matrix_fold[row['mutation_matrix_index']]=1
                location_matrix_fold[row['location_matrix_index']]=1

        mutation_matrix_bind = jnp.array(mutation_matrix_bind, dtype=jnp.float32)
        mutation_matrix_fold = jnp.array(mutation_matrix_fold, dtype=jnp.float32)
        location_matrix_bind = jnp.array(location_matrix_bind, dtype=jnp.float32)
        location_matrix_fold = jnp.array(location_matrix_fold, dtype=jnp.float32)

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

    with open('GRB2_SH3.pkl', 'wb') as fp:
        pickle.dump(data_dict, fp)
        print('dictionary saved successfully to file')

    return data_dict


if __name__ == '__main__':
    union_mode = False

    data_train_file = '/Users/pierredemetz/UCL_work/Crick/doubledeepms/Results//Data/mochi/GRB2-SH3/mochi__fit_tmodel_3state_sparse_dimsum128_subsample100p//dataset_train.txt'
    data_valid_file = '/Users/pierredemetz/UCL_work/Crick/doubledeepms/Results//Data/mochi/GRB2-SH3/mochi__fit_tmodel_3state_sparse_dimsum128_subsample100p//dataset_valid.txt'
    data_obs_file = '/Users/pierredemetz/UCL_work/Crick/doubledeepms/Results//Data/mochi/GRB2-SH3/mochi__fit_tmodel_3state_sparse_dimsum128_subsample100p//dataset_all.txt'

    data = load_model_data_complex({
    "train": data_train_file,
    "valid": data_valid_file,
    "obs": data_obs_file
    },
                                   union_mode)

    #pprint.pprint(data)
