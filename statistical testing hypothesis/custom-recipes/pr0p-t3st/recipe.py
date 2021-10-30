from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role,
)

import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from scipy import stats

#==================================================================================================\
# init
def set_prop_testing_config():
    """
    """
    recipe_config = get_recipe_config()
    test_col = recipe_config["column_to_test"]
    alpha_min = recipe_config["alpha_min"]
    alpha_max = recipe_config["alpha_max"]
    Nb_steps = recipe_config["Numb_steps"]
    prop_0 = recipe_config["prop_0"]
    alt_hypothesis = recipe_config["alt_hypothesis"]
    step_size = (alpha_max - alpha_min) / Nb_steps
    alphas = [np.round(alpha_min + k*step_size, 3) for k in range(Nb_steps + 1)]
    
    params = (test_col, Nb_steps, prop_0, alt_hypothesis, alphas)
    
    return params

#==================================================================================================\
# Load input data
def load_input_data(input_role_name = "input_dataset"):
    """
    """
    input_dataset_name = get_input_names_for_role(input_role_name)[0]
    input_dataset = dataiku.Dataset(input_dataset_name)
    df = input_dataset.get_dataframe()
    
    # Extract the numeric columns only to compute & evaluate the Confidence_interval
    numeric_columns = (df.dtypes != 'object').index
    df = df._get_numeric_data()
    df = df[[test_col]]
    
    return df

#==================================================================================================\
# Create a testing function
def prop_test(data, prop_0, alternative = 'equal', alpha = 0.05):
    """
        This function is used to implement the prop_test
        Input params:
            data (dataframe of boolean): dataframe restricted to the specified column to evaluate
            prop_0 (float) : values of the alternative testing
            alternative (strings) : 
                    + two.side [equal]: prop = prop_0
                    + less :            prop < prop_0
                    + greater :         prop > prop_0
            alpha (float) must be in (0, 1): significance level
        Returns
            dataframe that contains all info of the testing hypothesis
            such as
                    + Testing stastistic
                    + p_value
                    + conclusion
                    + so on.
    """
    count_event = float(data.sum())
    sample_size = len(data)
    sample_prop = count_event / sample_size
    confidence = 1 - alpha  # significane_level by confidence_level
    scaled = np.sqrt(sample_prop * (1 - sample_prop ) / sample_size )
    
    # Compute the testing statistics
    T_stats = (sample_prop - prop_0) / np.sqrt(sample_prop * (1 - sample_prop ) / sample_size )
    
    # argument
    if alternative in ["equal", "two.side"]:
        p_value = 2*min(stats.norm.cdf(T_stats), 1 - stats.norm.cdf(T_stats))
        radius = stats.norm.ppf(alpha / 2)*scaled
        conf_itv = (sample_prop - radius, sample_prop + radius)
        alt = "the true_proportion is not equal to {}".format(prop_0)
        null_hyp = "the true_proportion is equal to {}".format(prop_0)
    
    elif alternative == 'less':
        p_value = stats.norm.cdf(T_stats)
        radius = stats.norm.ppf(alpha)*scaled
        conf_itv = (0, sample_prop + radius)     
        alt = "the true_proportion is less than {}".format(prop_0)
        null_hyp = "the true_proportion >= {}".format(prop_0)
    
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(T_stats)
        radius = stats.norm.ppf(alpha)*scaled
        conf_itv = (sample_prop - radius, 1)    
        alt = "the true_proportion is greater than {}".format(prop_0)
        null_hyp = "the true_proportion <= {}".format(prop_0)
        
    # consclusion
    if p_value < alpha:
        conclusion = 'For confidence_level = {}% reject the hypothesis, so {}'.format(100*confidence, alt)
    else:
        conclusion = 'For confidence_level = {}% can not reject the hypothesis,'.format(100*confidence, null_hyp)
    
    # return
    res = pd.DataFrame({'significan_level(alpha)': alpha,
                        'confidence_interval': [conf_itv],
                        'conclusion': [conclusion],
                        'sample_proportion': sample_prop,
                        'testing statistics':T_stats,
                        'alternative': alt
                       })
    
    return res

#==================================================================================================\
# Loading all input params
test_col, Nb_steps, prop_0, alt_hypothesis, alphas = set_prop_testing_config()
df = load_input_data()

#==================================================================================================\
# Saving Output
prop_test_CI_df = prop_test(df, prop_0, alternative = alt_hypothesis, alpha = alphas[0])

#==================================================================================================\
# For this sample code, simply copy input to output
# Loof through all columns in dataset
for alpha in alphas[1:]:
    add_df = prop_test(df, prop_0, alternative = alt_hypothesis, alpha = alpha)
    prop_test_CI_df = pd.concat([prop_test_CI_df, add_df])
    
#==================================================================================================\
# Write recipe outputs
output_dataset_name = get_output_names_for_role("main_output")
output_dataset = dataiku.Dataset(output_dataset_name[0])
output_dataset.write_with_schema(prop_test_CI_df)