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
def set_variance2_testing_config():
    """
    """
    recipe_config = get_recipe_config()
    
    col_spl_A = recipe_config["col_sample_A"]
    col_spl_B = recipe_config["col_sample_B"]
    
    alpha_min = recipe_config["alpha_min"]
    alpha_max = recipe_config["alpha_max"]
    Nb_steps = recipe_config["Numb_steps"]
    alt_hypothesis = recipe_config["alt_hypothesis"]

    step_size = (alpha_max - alpha_min) / Nb_steps
    alphas = [np.round(alpha_min + k*step_size, 3) for k in range(Nb_steps + 1)]
    
    params = (col_spl_A, col_spl_B, alt_hypothesis, alphas)
    
    return params

# Read recipe inputs
def load_input_datasets(input_A_role_name, input_B_role_name):
    """
    
    """
    sample_A_name = get_input_names_for_role(input_A_role_name)[0]
    sample_B_name = get_input_names_for_role(input_B_role_name)[0]
    
    sample_A = dataiku.Dataset(sample_A_name)
    sample_B = dataiku.Dataset(sample_B_name)
    
    sample_A_df = sample_A.get_dataframe()
    sample_B_df = sample_B.get_dataframe()
    
    return sample_A_df[col_spl_A], sample_B_df[col_spl_B]
    
# Compute recipe outputs
def var_2_test(sample_A, sample_B, alternative, alpha = 0.05):
    """
    
    """
    
    # compute the sample sizes and confidence_level
    N_a = float(len(sample_A))
    N_b = float(len(sample_B))
    confidence = np.round(1 - alpha, 3)
    
    # compute the averages at 2 sample
    muy_a = float(sample_A.mean())
    muy_b = float(sample_B.mean())
    
    # compute the standard deviation each sample
    std_a = float(sample_A.std())
    std_b = float(sample_B.std())
    
    # Testing statistical
    T_stats = (std_a / std_b)**2
        
    # Argument w.r.t alternative
    if alternative in ["equal", "two.side"]:
        p_value = min(stats.f.cdf(T_stats, 
                                    dfn = N_a - 1, dfd = N_b - 1), 
                        1 - stats.f.cdf(T_stats, 
                                        dfn = N_b - 1, dfd = N_a - 1))
        low_rad = stats.f.ppf(alpha / 2, dfn = N_a - 1, dfd = N_b - 1)
        upp_rad = stats.f.ppf(alpha / 2, dfn = N_b - 1, dfd = N_a - 1)
        conf_itv = (T_stats / max(low_rad, upp_rad), T_stats / min(low_rad, upp_rad))
        alt = "the variances in these 2 samples is not equal."
        null_hyp = "(sigma_A)^2 = (sigma_B)^2"
        
    elif alternative == 'greater':
        p_value = 1 - stats.f.cdf(T_stats, 
                              dfn = N_a - 1, dfd = N_b - 1)
        low_rad = stats.f.ppf(alpha, dfn = N_a - 1, dfd = N_b - 1)
        upp_rad = stats.f.ppf(alpha, dfn = N_b - 1, dfd = N_a - 1)
        conf_itv = (T_stats / max(low_rad, upp_rad), '+inf')    
        alt = "(sigma_A)^2 > (sigma_B)^2"
        null_hyp = "(sigma_A)^2 <= (sigma_B)^2"
        
    elif alternative == 'less':
        p_value = stats.f.cdf(T_stats, 
                              dfn = N_a - 1, dfd = N_b - 1)
        low_rad = stats.f.ppf(alpha, dfn = N_a - 1, dfd = N_b - 1)
        upp_rad = stats.f.ppf(alpha, dfn = N_b - 1, dfd = N_a - 1)
        conf_itv = (0, T_stats / min(low_rad, upp_rad))     
        alt = "(sigma_A)^2 < (sigma_B)^2"
        null_hyp = "(sigma_A)^2 >= (sigma_B)^2"
    
    # consclusion
    if p_value < alpha:
        conclusion = 'For confidence_level = {}% reject the hypothesis, so {}'.format(100*confidence, alt)
    else:
        conclusion = 'For confidence_level = {}% can not reject the hypothesis,'.format(100*confidence, null_hyp)
    
    # return
    res = pd.DataFrame({'significan_level(alpha)': alpha,
                        'confidence_interval': [conf_itv],
                        'conclusion': [conclusion],
                        'p_value': p_value,
                        'variance_spl_A': std_a,
                        'variance_spl_B': std_b,
                        'testing statistics':T_stats,
                        'alternative': alt
                       })
    
    return res

col_spl_A, col_spl_B, alt_hypothesis, alphas = set_variance2_testing_config()
sample_A, sample_B = load_input_datasets("sample_A", "sample_B")
var_test_2samples_df = var_2_test(sample_A, sample_B, 
                                    alternative = alt_hypothesis,
                                    alpha = alphas[0]
                                   )

for alpha in alphas[1:]:
    add_df = var_2_test(sample_A, sample_B, 
                         alternative = alt_hypothesis, 
                         alpha = alpha)
    var_test_2samples_df = pd.concat([var_test_2samples_df, add_df])
    
# Write recipe outputs
output_dataset_name = get_output_names_for_role("main_output")
output_dataset = dataiku.Dataset(output_dataset_name[0])
output_dataset.write_with_schema(var_test_2samples_df)