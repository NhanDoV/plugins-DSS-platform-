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
def set_mean_testing_config():
    """
    """
    recipe_config = get_recipe_config()
    test_col = recipe_config["column_to_test"]
    alpha_min = recipe_config["alpha_min"]
    alpha_max = recipe_config["alpha_max"]
    Nb_steps = recipe_config["Numb_steps"]
    muy0 = recipe_config["muy0"]
    alt_hypothesis = recipe_config["alt_hypothesis"]
    # List of significance level
    step_size = (alpha_max - alpha_min) / Nb_steps
    alphas = [np.round(alpha_min + k*step_size, 3) for k in range(Nb_steps + 1)]
    
    return test_col, muy0, alt_hypothesis, alphas

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
def mean_test(data, muy0, alternative = 'equal', alpha = 0.95):
    """
        This function is used to create a confidence interval of two.sided hypothesis
        Input:
            data (1D array): the sample of the whole column that you want to evaluate
            confidence (float) : confidence_level, must be in (0, 1)
        Output:
            the dictionary that contains the info of
                - Confidence_Interval (tupple)
                - T_statistics (float)
                - Two-sided p-value.
    """
    # convert datapoint to float
    a = 1.0 * np.array(data)
    confidence = np.round(1 - alpha, 3)
    
    # Number of observations
    n = len(a)

    # Compute mean and standard_errors
    m, se = np.mean(a), stats.sem(a)
    
    # result of testing
    T_stat = ((m - muy0) / se)*np.sqrt(n)

    # compute the interval_radius
    if alternative in ['equal', "two.side"]:
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        conf_itv = (float(m - h), float(m + h))
        alt = "true mean (muy) is not equal to muy0 = {}".format(muy0)
        cnls = "true mean (muy) = muy0 = {}".format(muy0)
        p_val = 2*min(stats.t.cdf(T_stat, n-1), 1 - stats.t.cdf(T_stat, n-1))
        
    elif alternative == 'greater':
        h = se * stats.t.ppf(1 - confidence, n-1)
        conf_itv = (float(m - h), '+inf')
        alt = "true mean (muy) > muy0 = {}".format(muy0)
        cnls = "true mean (muy) <= muy0 = {}".format(muy0)
        p_val = 1 - stats.t.cdf(T_stat, n-1)
        
    elif alternative == 'less':
        h = se * stats.t.ppf(1 - confidence, n-1)
        conf_itv = ('-inf', float(m + h))
        alt = "true mean (muy) < muy0 = {}".format(muy0)
        cnls = "true mean (muy) >= muy0 = {}".format(muy0)
        p_val = stats.t.cdf(T_stat, n-1)
    
    # conclusion
    if p_val < alpha:
        kl = 'reject the hypothesis, {}'.format(alt)
    else:
        kl = 'can not reject the null hypothesis, so the {}'.format(cnls)
    
    # save all the output-results    
    dic_data = pd.DataFrame(
        {
            'alpha / confidence_level': [{'significance level':alpha, 'confidence_level': 1- alpha}],
            'Confidence_Interval': [conf_itv],
            'T_statistic': T_stat,
            'sample_mean': m,
            'alternative_hypothesis': alt,
            'p_value': p_val,
            'conclusion': "For confidence_level = {}%, we {}".format(100*confidence, kl)
          }
    )

    return dic_data

#==================================================================================================\
# Compute recipe outputs from inputs
test_col, muy0, alt_hypothesis, alphas = set_mean_testing_config()
df = load_input_data()
col_names = df.columns
mean_test_CI_df = mean_test(df, muy0, alternative = alt_hypothesis, alpha = 1 - alphas[0])

# For this sample code, simply copy input to output
# Loof through all columns in dataset
for alpha in alphas[1:]:
    add_df = mean_test(df, muy0, alternative = alt_hypothesis, alpha = 1 - alpha)
    mean_test_CI_df = pd.concat([mean_test_CI_df, add_df])

#==================================================================================================\    
# Write recipe outputs
output_dataset_name = get_output_names_for_role("main_output")
output_dataset = dataiku.Dataset(output_dataset_name[0])
output_dataset.write_with_schema(mean_test_CI_df)