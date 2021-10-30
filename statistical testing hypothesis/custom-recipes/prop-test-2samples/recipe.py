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
def set_prop_testing_config():
    """
    """
    recipe_config = get_recipe_config()
    
    col_spl_A = recipe_config["col_sample_A"]
    col_spl_B = recipe_config["col_sample_B"]
    
    alpha_min = recipe_config["alpha_min"]
    alpha_max = recipe_config["alpha_max"]
    Nb_steps = recipe_config["Numb_steps"]
    prop_0 = recipe_config["prop_0"]
    alt_hypothesis = recipe_config["alt_hypothesis"]
    step_size = (alpha_max - alpha_min) / Nb_steps
    alphas = [np.round(alpha_min + k*step_size, 3) for k in range(Nb_steps + 1)]
    
    params = (prop_0, col_spl_A, col_spl_B, Nb_steps, alt_hypothesis, alphas)
    
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
def prop_2_test(sample_A, sample_B, prop_0, alternative, alpha = 0.05):
    """
    
    """
    # loading vs transforming your input_params
    N_a = float(len(sample_A))
    N_b = float(len(sample_B))
    confidence = np.round(1 - alpha, 3)
    
    count_a = float(sample_A.sum())
    count_b = float(sample_B.sum())
    
    prop_A = count_a / N_a
    prop_B = count_b / N_b
    
    prop_est = (count_a + count_b) / (N_a + N_b)
    prop_dif = prop_A - prop_B
     
    # testing statistical
    T_stats = (prop_dif - prop_0) /np.sqrt(prop_est*(1 - prop_est)*(1 / N_a + 1/ N_b))
    pooled_scale = np.sqrt(prop_A*(1 - prop_A)/N_a + prop_B*(1 - prop_B) / N_b)
   
    # Argument w.r.t alternative
    if alternative in ["equal", "two.side"]:
        p_value = 2*min(stats.norm.cdf(T_stats), 1 - stats.norm.cdf(T_stats))
        radius = stats.norm.ppf(alpha / 2)*pooled_scale
        conf_itv = (prop_dif - radius, prop_dif + radius)
        alt = "the difference_proportions in these 2 samples is not equal to {}".format(prop_0)
        null_hyp = "prop_A - prop_B = {}".format(prop_0)
        
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(T_stats)
        radius = stats.norm.ppf(alpha)*pooled_scale
        conf_itv = (prop_dif - radius, 1)    
        alt = "the difference_proportions in these 2 samples is greater than {}".format(prop_0)
        null_hyp = "prop_A - prop_B <= {}".format(prop_0)
        
    elif alternative == 'less':
        p_value = stats.norm.cdf(T_stats)
        radius = stats.norm.ppf(alpha)*pooled_scale
        conf_itv = (0, prop_dif + radius)     
        alt = "the difference_proportions in these 2 samples is less than {}".format(prop_0)
        null_hyp = "prop_A - prop_B >= {}".format(prop_0)
    
    # consclusion
    if p_value < alpha:
        conclusion = 'For confidence_level = {}% reject the hypothesis, so {}'.format(100*confidence, alt)
    else:
        conclusion = 'For confidence_level = {}% can not reject the hypothesis,'.format(100*confidence, null_hyp)
    
    # return
    res = pd.DataFrame({'significan_level(alpha)': alpha,
                        'confidence_interval': [conf_itv],
                        'conclusion': [conclusion],
                        'proportion_spl_A': prop_A,
                        'proportion_spl_B': prop_B,
                        'testing statistics':T_stats,
                        'alternative': alt
                       })
    
    return res

prop_0, col_spl_A, col_spl_B, Nb_steps, alt_hypothesis, alphas = set_prop_testing_config()
sample_A, sample_B = load_input_datasets("sample_A", "sample_B")
prop_test_2samples_df = prop_2_test(sample_A, sample_B, prop_0, 
                                    alternative = alt_hypothesis,
                                    alpha = alphas[0]
                                   )
print("{}\n{}\n{}".format(100*"=", prop_test_2samples_df, 100*"="))
for alpha in alphas[1:]:
    add_df = prop_2_test(sample_A, sample_B, prop_0, 
                         alternative = alt_hypothesis, 
                         alpha = alpha)
    prop_test_2samples_df = pd.concat([prop_test_2samples_df, add_df])
    
# Write recipe outputs
output_dataset_name = get_output_names_for_role("main_output")
output_dataset = dataiku.Dataset(output_dataset_name[0])
output_dataset.write_with_schema(prop_test_2samples_df)