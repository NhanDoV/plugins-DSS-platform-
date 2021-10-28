#==================================================================================================\
# If exist, drop all the comment above
from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role,
)

# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from sklearn.decomposition import PCA

#==================================================================================================\
def set_pca_config():
    """
    """
    recipe_config = get_recipe_config()
    n_components = recipe_config["explanied_ratio"]
    cols = recipe_config["columns_to_keep"]
    svd_solver = recipe_config["svd_solver"]
    tolerance = recipe_config["tol"]
    whiten = recipe_config["whiten"]
    col_keep = [cols[k] for k in range(len(cols))]
    random_state = recipe_config["random_state"]
    
    return random_state, n_components, svd_solver, tolerance, whiten, col_keep

random_state, n_components, svd_solver, tolerance, whiten, col_keep = set_pca_config()

#==================================================================================================\
# Read recipe inputs
def load_input_data(input_role_name = "input_dataset"):
    """
    """
    input_dataset_name = get_input_names_for_role(input_role_name)[0]
    input_dataset = dataiku.Dataset(input_dataset_name)
    df = input_dataset.get_dataframe()
    df_col_keep = df[col_keep]
    df = df.drop(columns = col_keep)
    
    return df, df_col_keep

df, df_col_keep = load_input_data()
col_name = df.columns

#==================================================================================================\
def build_PCA_data(df):
    """
    """
    df = df._get_numeric_data()
    pca = PCA(
                n_components = n_components,
                tol = tolerance, 
                svd_solver = svd_solver,
                whiten = whiten, 
                random_state = random_state
             )
    pca.fit(df)
    data_pca = pca.transform(df)
    Nb_PCs = data_pca.shape[1]
    pca_columns = ['PC_' + str(idx + 1) for idx in range(Nb_PCs)]
    pca_data_df = pd.DataFrame(data = data_pca, columns = pca_columns)
    pca_data_df = pd.concat([df_col_keep, pca_data_df], axis = 1)
    
    return pca, pca_data_df

pca, pca_data_df = build_PCA_data(df)
    
#==================================================================================================\
pca_weights = pca.get_precision()
all_pca_col = ['PC_' + str(idx + 1) for idx in range(df.shape[1])]
pc_weights_df = pd.DataFrame(data = pca_weights, 
                             columns = all_pca_col, index = col_name)
pc_weights_df = pc_weights_df.reset_index().rename(columns = {'index':'column_weights'})

#==================================================================================================\
# Write recipe outputs
output_dataset_name = get_output_names_for_role("PCA_data")
pca_data = dataiku.Dataset(output_dataset_name[0])
pca_data.write_with_schema(pca_data_df)

output_dataset_name = get_output_names_for_role("PC_weights")
pc_weights = dataiku.Dataset(output_dataset_name[0])
pc_weights.write_with_schema(pc_weights_df)
