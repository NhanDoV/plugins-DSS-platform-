# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# If exist, drop all the comment above
from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role,
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from dataiku import pandasutils as pdu

# Read recipe inputs
input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
df = input_dataset.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
recipe_config = get_recipe_config()
date_col = recipe_config["date_col"] # 'Date'
target_col = recipe_config["target_col"] # 'Adj Close'
freqs = recipe_config["frequency"]
periods = recipe_config["periods"]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
series = df.set_index(date_col)[target_col].dropna()
result = seasonal_decompose(series.resample(freqs).sum(), freq=periods)

resid = result.resid.values
price = result.observed.values
dates = result.observed.index
seasn = result.seasonal.values
trend = result.trend.values

# Compute recipe outputs from inputs
this_df = pd.DataFrame({date_col: dates.ravel(),
                        'residual': resid.ravel(),
                        'seasonality': seasn.ravel(),
                        'trend':trend.ravel()
                       })
this_df[date_col] = pd.to_datetime(this_df[date_col])
that_df = result.observed.reset_index().drop(columns = date_col)
this_df = pd.concat([this_df, that_df], axis = 1)
this_df = this_df[[date_col, target_col, "residual", "trend", "seasonality"]]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
output_dataset_name = get_output_names_for_role("main_output")
saved_data = dataiku.Dataset(output_dataset_name[0])
saved_data.write_with_schema(this_df)
