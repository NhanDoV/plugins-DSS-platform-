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

# Read recipe inputs
input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
spam_df = input_dataset.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
recipe_config = get_recipe_config()
text_col = recipe_config["text_col"]
target_col = recipe_config["target_col"]
n_pc = recipe_config["n_pc"] 

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import re
def Tokenizer(str_input):
    ## 1. Remove url_link
    remove_url = re.compile(r'https?://\S+|www\.\S+').sub(r'', str_input)
    
    ## 2. Remove html_link
    remove_html = re.compile(r'<.*?>').sub(r'', remove_url)
    
    ## 3. Remove Emojis
    remove_emo = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE).sub(r'', remove_html)
    words = re.sub(r"[^A-Za-z\-]", " ", remove_emo).lower().split()    
        
    return words #' '.join(words)

def NLP_get_pca_data(data, text_col, target_col, n_pc = 200):
    """
    
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA, TruncatedSVD
    
    tfidf_vect = TfidfVectorizer(tokenizer = Tokenizer)
    y = data[[text_col, target_col]]

    pca = TruncatedSVD(n_pc)
    
    tfidf_X = tfidf_vect.fit(data[text_col]).transform(data[text_col])
    pca.fit_transform(tfidf_X)
    
    X_pca = pd.DataFrame(data = pca.fit_transform(tfidf_X),
                         columns = ['PC_{}'.format(k) for k in range(n_pc)]
                        )
    
    print(X_pca.shape, pca.explained_variance_ratio_.sum())
    
    return pd.concat([y, X_pca], axis = 1)

pca_term_freq_df = NLP_get_pca_data(spam_df, 'text', 'target', 100)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
output_dataset_name = get_output_names_for_role("main_output")
saved_data = dataiku.Dataset(output_dataset_name[0])
saved_data.write_with_schema(pca_term_freq_df)
