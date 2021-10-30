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
def set_distribution_testing_config():
    """
    """
    recipe_config = get_recipe_config()
    distribution_name = recipe_config["distribution_name"]
    alpha_min = recipe_config["alpha_min"]
    alpha_max = recipe_config["alpha_max"]
    test_col = recipe_config["col_to_test"]
    Nb_steps = recipe_config["Numb_steps"]
    alt_hypothesis = recipe_config["alt_hypothesis"]

    step_size = (alpha_max - alpha_min) / Nb_steps
    alphas = [np.round(alpha_min + k*step_size, 3) for k in range(Nb_steps + 1)]
    
    params = test_col, distribution_name, alt_hypothesis, alphas
    
    return recipe_config, params

#==================================================================================================\
def load_input_datasets(input_role_name):
    """
    
    """
    sample_name = get_input_names_for_role(input_role_name)[0]
    sample_data = dataiku.Dataset(sample_name)
    sample_df = sample_data.get_dataframe()
    
    return sample_df

#==================================================================================================\
def distribution_test(data):
    """
    
    """
    config, params = set_distribution_testing_config()
    test_col, cdf, alternative, alphas = params
    
    data = data[test_col]
    
    # compute the statistical_testing and p_value
    # 1. Normal(muy, sigma^2)
    if cdf == 'norm':
        name = 'Normal'
        # Loading all params_distribution
        norm_muy = config["norm_muy"]   
        norm_sig = config["norm_sigma^2"]
        T_stats, p_val = stats.kstest(data, alternative = alternative,
                                      cdf = stats.norm(loc = norm_muy, 
                                                       scale = norm_sig).cdf)

    # 2. Poisson (lambda, loc)    
    elif cdf == 'poisson':      
        name = 'Poisson'
        pois_lamb = config["pois_muy"]
        pois_loc = config["pois_loc"]        
        T_stats, p_val = stats.kstest(data, alternative = alternative,
                                      cdf = stats.poisson(muy = pois_lamb, 
                                                          loc = pois_loc).cdf)
    # 3. Student(df)
    elif cdf == 't':      
        name = 'Student'
        st_deg = config["St_deg"]
        T_stats, p_val = stats.kstest(data, alternative = alternative,
                                      cdf = stats.t(df = int(st_deg), loc = 0).cdf)
        
    # 4. Fisher(df1, df2)
    elif cdf == 'f':
        name = 'Fisher'
        f_dfd = config["F_deg_denom"]
        f_dfn = config["F_deg_numer"]
        T_stats, p_val = stats.kstest(data, alternative = alternative,
                                      cdf = stats.f(dfn = f_dfn,
                                                    dfd = f_dfd).cdf)
            
    # 5. Binomial
    elif cdf =='binom':
        name = 'Binomial'
        binom_n = config["Binom_size"]
        binom_p = config["Binom_prob"] 
        T_stats, p_val = stats.kstest(data, alternative = alternative,
                                      cdf = stats.binom(n = binom_n,
                                                        p = binom_p).cdf)
    conclusions = []
    
    for alpha in alphas:
        if alpha < p_val:
            conclusions.append('For {}% confidence_level, we reject null-hypothesis. This data is not {} distribution'.format(100*(1-alpha), name))
        else:
            conclusions.append('For {}% confidence_level, this data is the {} distribution!'.format(100*(1-alpha), name))

    res = pd.DataFrame({'alpha': alphas,
                        'T_stats': T_stats,
                        'conclusion':conclusions
                       })
            
    return res

#==================================================================================================\
recipe_config, params = set_distribution_testing_config()
data = load_input_datasets("input_dataset")
dist_test_df = distribution_test(data)

# Write recipe outputs
output_dataset_name = get_output_names_for_role("main_output")
output_dataset = dataiku.Dataset(output_dataset_name[0])
output_dataset.write_with_schema(dist_test_df)