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
import re
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Read recipe inputs
input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
spam_df = input_dataset.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
recipe_config = get_recipe_config()
text_col = recipe_config["text_col"]
row_begin = recipe_config["row_begin"] 
row_end = recipe_config["row_end"] 
is_clean = recipe_config["is_cleaned"]
filter_df = spam_df.loc[row_begin: row_end]

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
        
    return ' '.join(words)

def textdata_to_countvect(data, text_column, is_cleaned = is_clean):
    """
        Parameters
        --------------------
            - data (dataframe)
            - text_column : the column contained text in your dataframe
        Returns
        --------------------
            - dataset of text and its count_vectorizer
        See also
        --------------------
            - view_word_freq(word_list)
    """
    data = data.rename(columns = {text_column: 'your_text'})
    if is_cleaned == 0:
        corpus = data['your_text'].values.tolist()
    else:
        corpus = [Tokenizer(data['your_text'][k]) for k in range(len(data))]
    countvec_df = view_word_freq(corpus, show_info = False).reset_index().rename(columns = {'index': 'your_text'})

    return countvec_df

def view_word_freq(word_list, show_info = True):
    """
        -------------------------------------------------------------------------------------------------------
          This function is used to view the frequencies (count) of word in each documents and the whole dataset
        -------------------------------------------------------------------------------------------------------
          Parameters:
        ----------------------
              word_list (list): list of sentences
          Returns:
        ----------------------
              dataframe contain the column of word_vocabularies with the indexes be the sentences in "word_list".
          Note. 
        ----------------------
              To save the results as a parameter, the number of words and sentences in your sample must be reasonable        
        |======================================================================================================
        | Example.
        | >>> corpus = ['this is the first document',
        |               'this document is the second document',
        |               'and this is the third one',
        |               'is this the first document?',
        |               'this Document is not yours..'
        |              ]
        | >>> view_word_freq(corpus)
        |*====================================================================================================
        ||There are 5 sentences in this corpus.
        ||====================================================================================================
        |*The number of the different words is 11, and ... they are:
        ||====================================================================================================
        |*	 1: and,
        |*	 2: document,
        |*	 3: first,
        |*	 4: is,
        |*	 5: not,
        |*	 6: one,
        |*	 7: second,
        |*	 8: the,
        |*	 9: third,
        |*	 10: this,
        |*	 11: yours,
        |  	 			 	and	document	first	is	not	one	second	the	third	this	yours
        | this is the first document		0	1		1	1	0	0	0	1	0	1	0
        | this document is the second document	0	2		0	1	0	0	1	1	0	1	0
        | and this is the third one 	 	1	0		0	1	0	1	0	1	1	1	0
        | is this the first document?	 	0	1		1	1	0	0	0	1	0	1	0
        | this Document is not yours..	 	0	1		0	1	1	0	0	0	0	1	1
        |======================================================================================================
    """
    cvect = CountVectorizer()
    X = cvect.fit_transform(word_list)

    if show_info == 1:
        print("*{}\n|There are {} sentences in this corpus.\n|{}".format(100*"=", X.shape[0], 100*"="))
        print("|The number of the different words is {}, and ... they are:\n|{}".format(X.shape[1], 100*"="))
        for idx, word in enumerate(cvect.get_feature_names()):
            print("*\t {}: {},".format(idx + 1, word))

    else:
        pass
    return pd.DataFrame(data = X.toarray(),
                        columns = cvect.get_feature_names(),
                        index = word_list 
                        )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
count_vct_df = textdata_to_countvect(filter_df, text_col)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
# Write recipe outputs
output_dataset_name = get_output_names_for_role("main_output")
saved_data = dataiku.Dataset(output_dataset_name[0])
saved_data.write_with_schema(count_vct_df)
