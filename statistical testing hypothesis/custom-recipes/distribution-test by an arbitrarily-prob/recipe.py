#############################
# Your original recipe
#############################
# -*- coding: utf-8 -*-
from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role,
)

import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from scipy import stats

# Read the input_params
#==================================================================================================\
# init
def set_dist2_testing_config():
    """
    
    """
    recipe_config = get_recipe_config()
    alpha_min = recipe_config["alpha_min"]
    alpha_max = recipe_config["alpha_max"]
    theo_prob_col = recipe_config["theo_prob_col"]
    emp_freq_col = recipe_config["emp_freq_col"]
    Nb_steps = recipe_config["Numb_steps"]
    step_size = (alpha_max - alpha_min) / Nb_steps
    alphas = [np.round(alpha_min + k*step_size, 3) for k in range(Nb_steps + 1)]
    
    params = theo_prob_col, emp_freq_col, alphas
    
    return params

#==================================================================================================\
def load_input_datasets(input_role_name):
    """
    
    """
    sample_name = get_input_names_for_role(input_role_name)[0]
    sample_data = dataiku.Dataset(sample_name)
    sample_df = sample_data.get_dataframe()
    
    return sample_df

#==================================================================================================\
def dist2_test(data):
    """
    
    """
    theo_prob_col, emp_freq_col, alphas = set_dist2_testing_config()
    n_cases = len(data)
    empirical = data[emp_freq_col]
    theoritic = data[theo_prob_col]
    N_obs = float(sum(empirical))    
    theo_freq = N_obs * theoritic
    deg_fredm = n_cases - 2
    Q = float(((theoritic - empirical)**2 / theoritic).sum())
    T_stats, p_val = stats.chisquare(empirical, theoritic)
    
    conclusions = []    
    for alpha in alphas:
        if alpha < p_val:
            conclusions.append('For confidence_level = {}%, reject H0, your empirical sample didnt fit these probabilities'.format(100*(1 - alpha)))
        else:
            conclusions.append('For confidence_level = {}%, reject H0, your empirical sample fit to these probabilities'.format(100*(1 - alpha)))
            
    res = pd.DataFrame({'alpha': alphas,
                        'T_stats': Q,
                        'p_value': p_val,
                        'conclusion': conclusions
                       })
    return res
    
#theo_prob_col, emp_freq_col, alphas = set_dist2_testing_config()
data = load_input_datasets("input_dataset") 
dist2_test_df = dist2_test(data)

#==================================================================================================\
# Write recipe outputs
output_dataset_name = get_output_names_for_role("main_output")
output_dataset = dataiku.Dataset(output_dataset_name[0])
output_dataset.write_with_schema(dist2_test_df)