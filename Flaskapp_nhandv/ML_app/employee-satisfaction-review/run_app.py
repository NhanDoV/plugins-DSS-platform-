from pickletools import int4
import joblib, pickle, json, re, csv, string
import pandas as pd, numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from flask import Flask, render_template, request
from pyvi import ViTokenizer, ViPosTagger, ViUtils
from datetime import datetime

app = Flask(__name__)
today = datetime.today().__str__()[:19]

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

def tokenize(text):
    pass

@app.route('/')
def index():
    return render_template('index.html', today = today)

ml_model = joblib.load('models/my_model.h5')
vect_tf = joblib.load('models/viet_vect_fpt.pkl')

categories = ['BUL', 'IT (mạng, nhân viên, hạ tầng dữ liệu, etc)', 'HR', 'PM', 'lead', 'dong_nghiep', 'luong_thuong', 
              'CSVC (canteen, workspace, equipment, etc)', 'tang_ca', 
              'nghi_phep', 'thoi_viec', 'bao_hiem', 'ho_tro_# (di_lai, cong_doan, tro_cap#)', 
              'cong_viec', 'dao_tao']
eval_list = ["Không có đánh giá", "Rất ko hài lòng 💩😠", "Chưa hài lòng 😥", 
            "Bình thường 😆", "Hài lòng 🥰", "Rất dzừa lònggg! ❤️💯"]

@app.route('/', methods=['POST'])
def score():
    org_text = request.form['text']
    text = clean_text(org_text)
    cleaned_text = ' '.join(ViPosTagger.postagging(ViTokenizer.tokenize(text))[0])
    clf_vect = vect_tf.transform([cleaned_text])
    preds = ml_model.predict(clf_vect)[0]
    y_nghia = [eval_list[int(preds[idx])] for idx in range(len(categories))]
    table = pd.DataFrame(data = preds, index=categories).reset_index().rename(columns={0: 'rating_level', 
                                                                                      'index': 'hạng mục đánh giá'})
    table['rating_level'] = table['rating_level'].astype(int)                                                                                      
    table['ý_nghĩa'] = y_nghia

    print(f'your text : {org_text}')

    return(render_template('index.html', 
                            tables = [table.to_html(classes="data")],
                            titles = table.columns.values,
                            your_inp = org_text,
                            today = today
                            )
                        )    

if __name__ == "__main__":
    app.run(port='8080', threaded=False, debug=True)