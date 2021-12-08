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
from dataiku import pandasutils as pdu
from collections import defaultdict
from wordcloud import STOPWORDS

# Read recipe inputs
input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
spam_df = input_dataset.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
recipe_config = get_recipe_config()
top_N = recipe_config["top_words"]
text_col = recipe_config["text_col"]
target_col = recipe_config["target_col"]
ngrams_type = recipe_config["ngrams_type"]
pst_meant = recipe_config["pst_meant"]
ngrams_type = recipe_config["ngrams_type"]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def generate_ngrams(text, n_gram = 1):
    """
        This function create the word_list as N-grams in your input_text
        --------------------------
        Parameters:
        --------------------------
            - text (str): input strings
            - ngram (int) must be 1 (unigram), 2(bi-grams), 3(tri-grams)
        Example
        --------------------------
        >>> init_text = "\U0001F600-\U0001F64F heallo, haev a godo jbo, todyao is 29Jun2021 !PlesAe Visit https://google.com.vn and contact abc2083181@yahoo.com"
            ['😀-🙏',
             'heallo,',
             'haev',
             'godo',
             'jbo,',
             'todyao',
             '29jun2021',
             '!plesae',
             'visit',
             'https://google.com.vn',
             'contact',
             'abc2083181@yahoo.com']
    """
    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]


def get_N_grams(data, text_col, target_col, positive_value, n_gram):
    """
        This function returns 2 datasets of N_grams
            - The first one contain the positive value meaning from the target-column
            - The last one .............negative
        Parameters
        ---------------------------
            - data (dataframe)
            - target_col : the target column
            - positive_value : the value in the target column that had a positive-meaning
            - n_gram (int)
    """     
    mask = (data[target_col] == positive_value)
    positive_N_grams = defaultdict(int)
    negative_N_grams = defaultdict(int)

    for text in data[mask][text_col]:
        for word in generate_ngrams(text, n_gram=n_gram):
            positive_N_grams[word] += 1

    for text in data[~mask][text_col]:
        for word in generate_ngrams(text, n_gram=n_gram):
            negative_N_grams[word] += 1

    df_positive_N_grams = pd.DataFrame(sorted(positive_N_grams.items(), key=lambda x: x[1])[::-1])
    df_negative_N_grams = pd.DataFrame(sorted(negative_N_grams.items(), key=lambda x: x[1])[::-1])
    
    return df_positive_N_grams, df_negative_N_grams

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
n_gram_df_0, n_gram_df_1 =  get_N_grams(spam_df, text_col, target_col, pst_meant, ngrams_type)
n_gram_df_0 = n_gram_df_0.rename(columns = {0: 'n_grams', 1: 'n_grams_count'}).head(top_N)
n_gram_df_1 = n_gram_df_1.rename(columns = {0: 'n_grams', 1: 'n_grams_count'}).head(top_N)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
output_dataset_name = get_output_names_for_role("negative_case")
negative_case = dataiku.Dataset(output_dataset_name[0])
negative_case.write_with_schema(n_gram_df_1)

output_dataset_name = get_output_names_for_role("positive case")
positive case = dataiku.Dataset(output_dataset_name[0])
positive case.write_with_schema(n_gram_df_0)
