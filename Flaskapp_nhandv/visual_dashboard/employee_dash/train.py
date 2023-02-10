import pandas as pd, numpy as np
from pyvi import ViTokenizer, ViPosTagger, ViUtils
import os, nltk, time, pickle, string, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

warnings.filterwarnings("ignore")
cd = os.getcwd()
print(f"The current directory was {cd}")

#nltk.download(['stopwords', 'punkt', 'wordnet'])
print('Downdoad nltk pack successfully!!!')

def count_cate(df):
    """
    """
    categories = [col for col in df.columns if col != 'review']
    rate_df = pd.DataFrame({})
    for cate in categories:
        rate_df = pd.concat([rate_df, df.groupby(cate).count()], axis=1)[['review']].sort_index().fillna(0)#
        #rate_df = rate_df.rename(columns = {'review': cate})

    rate_df.columns = categories
    return rate_df.astype(int)

def clean_text(txt):

    # lowercase
    txt = txt.lower()
    txt = txt.replace('vs', 'và').replace('ko', 'không').replace('project', 'dự án').replace('proj', 'dự án').replace('bth', 'bình thường')
    txt = txt.replace('leader', 'lead').replace('711', 'Seven Eleven')
    # remove punctuation
    for punct in string.punctuation:
        if punct != '_':
            txt = txt.replace(punct, '')

    for digit in range(10):
        txt = txt.replace(str(digit), '')

    return txt
    
data_path = "data/fpt_reviews_simulation_data.csv"
df = pd.read_csv(f"{cd}//{data_path}")
print("Your dataset has been totally loaded")

print(f"{100*'='}\n Starting word pre-processing")
stri = df['review'].apply(lambda x: ' '.join(ViPosTagger.postagging(ViTokenizer.tokenize(x))[0])).values.tolist()
word_list = [clean_text(text) for text in stri]
tfvect = TfidfVectorizer()
xvect = tfvect.fit_transform(word_list)

print(f"{100*'='}\n Initializing model objects")
categories = [col for col in df.columns if col != 'review']
clf_model = GradientBoostingClassifier()
multi_model = MultiOutputClassifier(clf_model)

print(f"{100*'='}\n Starting vectorization")
tfvect = TfidfVectorizer()
xvect = tfvect.fit_transform(word_list)

print(f"{100*'='}\n Starting training")
multi_model.fit(xvect, df[categories])

pickle.dump(tfvect, open(f"{cd}//models//viet_vect_fpt.pkl", 'wb'))
pickle.dump(multi_model, open(f"{cd}//models//my_model.h5", 'wb'))

print("Models & package saved successfully")
print("Finish... cáo từ & chim cút")