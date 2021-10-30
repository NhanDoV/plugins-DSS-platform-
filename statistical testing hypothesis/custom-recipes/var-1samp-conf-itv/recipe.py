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
def set_var_testing_config():
    """
    """
    recipe_config = get_recipe_config()
    test_col = recipe_config["column_to_test"]
    alpha_min = recipe_config["alpha_min"]
    alpha_max = recipe_config["alpha_max"]
    Nb_steps = recipe_config["Numb_steps"]
    sigma0 = recipe_config["sigma0"]
    alt_hypothesis = recipe_config["alt_hypothesis"]

    # List of significance level
    step_size = (alpha_max - alpha_min) / Nb_steps
    alphas = [np.round(alpha_min + k*step_size, 3) for k in range(Nb_steps + 1)]

    return test_col, sigma0, alt_hypothesis, alphas

#==================================================================================================\
# Read recipe inputs
def read_input_data(input_role_name = "input_dataset"):
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

def var_test(data, sigma0, alternative = 'equal', alpha = 0.05):
    """
        This function is used to create a confidence interval of two.sided hypothesis
        Input:
            data (1D array): the sample of the whole column that you want to evaluate
            alpha (float) : significan_level, must be in (0, 1)
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
    N = len(a)

    # Compute mean and standard_errors
    m, se = np.mean(a), np.std(a)

    # result of testing
    T_stats = (N - 1)/(se**2 / sigma0**2)
    h = (N-1)*se
    
    # compute the interval_radius
    if alternative == 'equal':        
        low = min(stats.chi2.ppf((1 + confidence) / 2., N-1), stats.chi2.ppf((1 - confidence) / 2., N-1))
        upp = max(stats.chi2.ppf((1 + confidence) / 2., N-1), stats.chi2.ppf((1 - confidence) / 2., N-1))
        conf_itv = (float(h / upp), float(h / low))
        alt = "true variance (sigma) is not equal to sigma0 = {}".format(sigma0)
        cnls = "true variance (sigma) = sigma0 = {}".format(sigma0)
        p_val = 2*min(stats.chi2.cdf(T_stats, N - 1), 1 - stats.chi2.cdf(T_stats, N - 1))
        
    elif alternative == 'greater':
        low = min(stats.chi2.ppf((1 - confidence), N-1), stats.chi2.ppf(confidence, N-1))
        upp = max(stats.chi2.ppf((1 - confidence), N-1), stats.chi2.ppf(confidence, N-1))
        conf_itv = (float(h / upp), '+inf')
        alt = "true variance (sigma) > sigma0 = {}".format(sigma0)
        cnls = "true variance (sigma) <= sigma0 = {}".format(sigma0)
        p_val = 1 - stats.chi2.cdf(T_stats, N - 1)
        
    elif alternative == 'less':
        low = min(stats.chi2.ppf((1 - confidence), N-1), stats.chi2.ppf(confidence, N-1))
        upp = max(stats.chi2.ppf((1 - confidence), N-1), stats.chi2.ppf(confidence, N-1))
        conf_itv = ('0', float(h / low))
        alt = "true variance (sigma) < sigma0 = {}".format(sigma0)
        cnls = "true variance (sigma) >= sigma0 = {}".format(sigma0)
        p_val = stats.chi2.cdf(T_stats, N - 1)
    
    # conclusion
    if p_val < alpha:
        kl = 'Reject the hypothesis, {}'.format(alt)
    else:
        kl = 'can not reject the null hypothesis, so the {}'.format(cnls)
    
    # save all the output-results    
    dic_data = pd.DataFrame(
        {
            'alpha / confidence_level': [{'significance level': alpha, 'confidence_level': 1- alpha}],
            'Confidence_Interval': [conf_itv],
            'T_statistic': T_stats,
            'sample_variance': se,
            'alternative_hypothesis': alt,
            'p_value': p_val,
            'conclusion': kl
          }
    )

    return dic_data

# Save the column_names for the later iteration
test_col, sigma0, alt_hypothesis, alphas = set_var_testing_config()
df = read_input_data()
col_names = df.columns

# Compute recipe outputs from inputs
var_test_CI_df = var_test(df, sigma0, alternative = alt_hypothesis, alpha = alphas[0])

# For this sample code, simply copy input to output
# Loof through all columns in dataset
for alpha in alphas[1:]:
    add_df = var_test(df, sigma0, alternative = alt_hypothesis, alpha = alpha)
    var_test_CI_df = pd.concat([var_test_CI_df, add_df])

# Write recipe outputs
output_dataset_name = get_output_names_for_role("main_output")
output_dataset = dataiku.Dataset(output_dataset_name[0])
output_dataset.write_with_schema(var_test_CI_df)