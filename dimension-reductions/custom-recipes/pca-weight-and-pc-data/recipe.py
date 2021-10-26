# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
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

recipe_config = get_recipe_config()
will_be_drop_or_not = recipe_config["will be drop or not"]
n_components = recipe_config["explanied_ratio"]
cols = recipe_config["columns_to_keep"]
print(100*"=")
col_keep = [cols[k] for k in range(len(cols))]
print(col_keep) #
print(100*"=")


# Read recipe inputs
input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
df = input_dataset.get_dataframe()
df = df.drop(columns = col_keep)
df_col_keep = df[[col_keep]]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = df._get_numeric_data()
col_name = df.columns
pca = PCA(n_components)
pca.fit(df)
data_pca = pca.transform(df)
Nb_PCs = data_pca.shape[1]
pca_columns = ['PC_' + str(idx + 1) for idx in range(Nb_PCs)]
pca_data_df = pd.DataFrame(data = data_pca, columns = pca_columns)
pca_data_df = pd.concat([df_col_keep, pca_data_df], axis = 1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pca_weights = pca.get_precision()
all_pca_col = ['PC_' + str(idx + 1) for idx in range(df.shape[1])]
pc_weights_df = pd.DataFrame(data = pca_weights, 
                             columns = all_pca_col, index = col_name)
pc_weights_df = pc_weights_df.reset_index().rename(columns = {'index':'column_weights'})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
output_dataset_name = get_output_names_for_role("PCA_data")
pca_data = dataiku.Dataset(output_dataset_name[0])
pca_data.write_with_schema(pca_data_df)

output_dataset_name = get_output_names_for_role("PC_weights")
pc_weights = dataiku.Dataset(output_dataset_name[0])
pc_weights.write_with_schema(pc_weights_df)
