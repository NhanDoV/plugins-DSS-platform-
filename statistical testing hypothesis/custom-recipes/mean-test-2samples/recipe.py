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
def set_mean2_testing_config():
    """
    """
    recipe_config = get_recipe_config()
    
    col_spl_A = recipe_config["col_sample_A"]
    col_spl_B = recipe_config["col_sample_B"]
    
    alpha_min = recipe_config["alpha_min"]
    alpha_max = recipe_config["alpha_max"]
    Nb_steps = recipe_config["Numb_steps"]
    muy_0 = recipe_config["muy_0"]
    alt_hypothesis = recipe_config["alt_hypothesis"]
    is_var_equal = recipe_config["is_var_equal"]

    step_size = (alpha_max - alpha_min) / Nb_steps
    alphas = [np.round(alpha_min + k*step_size, 3) for k in range(Nb_steps + 1)]
    
    params = (muy_0, is_var_equal, col_spl_A, col_spl_B, Nb_steps, alt_hypothesis, alphas)
    
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
def mean_2_test(sample_A, sample_B, muy_0, alternative, alpha = 0.05):
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
    
    # compute the common variance with respect to "is_var_equal"
    if is_var_equal != 1:
        # degree of freedom
        deg_frd = N_a + N_b - 2
        
        # common variance
        cmn_var = ((N_a - 1)*std_a**2 + (N_b - 1)*std_b**2) / deg_frd
      
        # T_statistics
        T_stats = (muy_a - muy_b - muy_0) / np.sqrt(cmn_var*(1/N_a + 1/N_b))
    
    else:
        cmn_var = (std_a**2 / N_a + std_b**2 / N_b)
        T_stats = (muy_a - muy_b - muy_0) / np.sqrt(cmn_var)
        deg_frd = cmn_var**2 / ((std_a**2 / N_a)/(N_a - 1) + (std_b**2 / N_b)/(N_b - 1))
        
    # Argument w.r.t alternative
    if alternative in ["equal", "two.side"]:
        p_value = 2*min(stats.t.cdf(T_stats, df = deg_frd), 1 - stats.t.cdf(T_stats, df = deg_frd))
        radius = stats.t.ppf(alpha / 2, df = deg_frd)*np.sqrt(cmn_var)
        conf_itv = ((muy_a - muy_b) - radius, (muy_a - muy_b) + radius)
        alt = "the difference_means in these 2 samples is not equal to {}".format(muy_0)
        null_hyp = "muy_A - muy_B = {}".format(muy_0)
        
    elif alternative == 'greater':
        p_value = 1 - stats.t.cdf(T_stats, df = deg_frd)
        radius = stats.t.ppf(alpha, df = deg_frd)**np.sqrt(cmn_var)
        conf_itv = ((muy_a - muy_b) - radius, 1)    
        alt = "the difference_means in these 2 samples is greater than {}".format(muy_0)
        null_hyp = "muy_A - muy_B <= {}".format(muy_0)
        
    elif alternative == 'less':
        p_value = stats.t.cdf(T_stats, df = deg_frd)
        radius = stats.t.ppf(alpha, df = deg_frd)**np.sqrt(cmn_var)
        conf_itv = (0, (muy_a - muy_b) + radius)     
        alt = "the difference_means in these 2 samples is less than {}".format(muy_0)
        null_hyp = "muy_A - muy_B >= {}".format(muy_0)
    
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
                        'avg_spl_A': muy_a,
                        'avg_spl_B': muy_b,
                        'testing statistics':T_stats,
                        'alternative': alt
                       })
    
    return res

muy_0, is_var_equal, col_spl_A, col_spl_B, Nb_steps, alt_hypothesis, alphas = set_mean2_testing_config()
sample_A, sample_B = load_input_datasets("sample_A", "sample_B")
mean_test_2samples_df = mean_2_test(sample_A, sample_B, muy_0, 
                                    alternative = alt_hypothesis,
                                    alpha = alphas[0]
                                   )

for alpha in alphas[1:]:
    add_df = mean_2_test(sample_A, sample_B, muy_0, 
                         alternative = alt_hypothesis, 
                         alpha = alpha)
    mean_test_2samples_df = pd.concat([mean_test_2samples_df, add_df])
    
# Write recipe outputs
output_dataset_name = get_output_names_for_role("main_output")
output_dataset = dataiku.Dataset(output_dataset_name[0])
output_dataset.write_with_schema(mean_test_2samples_df)