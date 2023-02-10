from pickletools import int4
import joblib, pickle, json, re, csv, string, os
import pandas as pd, numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from flask import Flask, render_template, request
from pyvi import ViTokenizer, ViPosTagger, ViUtils
from datetime import datetime
import json, plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
cd = os.getcwd()
today = datetime.today().__str__()[:19]

def rating_group(df, colname):
    return df.groupby(colname).count()[['review']].reset_index().rename(columns={colname: 'rating', 'review': f'count_{colname}'}).astype(int)

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

#@app.route('/enhanced')
@app.route('/review')
def review():
    df = pd.read_csv("data/fpt_reviews_simulation_data.csv").fillna(0)
    df[df.drop(columns='review').columns.tolist()] = df[df.drop(columns='review').columns.tolist()].astype(int)
    nrow, ncol = df.shape
    return render_template('review.html', df_view = df, nrow=nrow, ncol=ncol)

@app.route('/visualize')
def visualize():
    df = pd.read_csv("data/fpt_reviews_simulation_data.csv")
    figures = []
    ser = rating_group(df, 'BUL').merge(rating_group(df, 'IT'), on='rating')
    for col in df.drop(columns=['review', 'BUL', 'IT']).columns:
        ser = ser.merge(rating_group(df, col), on='rating', how='outer')

    fig_1 = px.bar(ser[ser['rating'] !=0 ], x='rating', y=[f"count_{col}" for col in df.drop(columns='review').columns], 
                        barmode="group", width=1200, height=600)
    layout_1 = dict(title = "Count plot for all evaluated_aspect <br> wrt rating_levels",
                    xaxis = dict(title = "rating level"),
                    yaxis = dict(title = "count")
                    )

    rate_df = pd.DataFrame(columns=['pos_rate', 'neg_rate', 'n_mentioned'])
    for field in df.drop(columns='review').columns:
        sub_df = df[df[field]!=0]
        rate_df.loc[field, 'pos_rate'] = round(100*len(sub_df[sub_df[field]>3]) / len(sub_df), 2)
        rate_df.loc[field, 'neg_rate'] = round(100*len(sub_df[sub_df[field]<3]) / len(sub_df), 2)
        rate_df.loc[field, 'n_mentioned'] = len(sub_df)

    rate_df = rate_df.astype(float).reset_index().rename(columns={'index':'fields'})
    rate_df = rate_df.sort_values(by=['pos_rate', 'neg_rate'], ascending=[False, True])
    fig_2 = px.bar(rate_df, x='fields', y=['pos_rate', 'neg_rate'], 
                     color_discrete_sequence =['cyan', 'magenta'],
                     hover_data=['n_mentioned'], barmode="group", width=600)

    # Fig 3
    figures3 = make_subplots(rows=5, cols=3)
    for k, field in enumerate(df.drop(columns='review').columns):
        sub_df = df[df[field]!=0]
        df_pos = sub_df[sub_df[field]>3]
        df_neg = sub_df[sub_df[field]<3]

        stri_pos = df_pos['review'].apply(lambda x: ' '.join(ViPosTagger.postagging(ViTokenizer.tokenize(x))[0])).values.tolist()
        word_list_pos = [clean_text(text) for text in stri_pos]

        stri_neg = df_neg['review'].apply(lambda x: ' '.join(ViPosTagger.postagging(ViTokenizer.tokenize(x))[0])).values.tolist()
        word_list_neg = [clean_text(text) for text in stri_neg]

        tfvect = CountVectorizer()
        xvect_pos = tfvect.fit_transform(word_list_pos)
        sdf_pos = pd.DataFrame(data = xvect_pos.toarray(), columns= tfvect.get_feature_names_out()).sum().sort_values(ascending=False).head(10).reset_index()

        tfvect = CountVectorizer()
        xvect_neg = tfvect.fit_transform(word_list_neg)
        sdf_neg = pd.DataFrame(data = xvect_neg.toarray(), columns= tfvect.get_feature_names_out()).sum().sort_values(ascending=False).head(10).reset_index()

        sdf_pos['flag'] = 'pos'
        sdf_pos[field] = (sdf_pos.index + 1).map(lambda x: f"top_{x}_word")
        sdf_neg['flag'] = 'neg'
        sdf_neg[field] = (sdf_neg.index + 1).map(lambda x: f"top_{x}_word")
        sdf = pd.concat([sdf_pos.rename(columns={'index': 'text', 0:'count'}), sdf_neg.rename(columns={'index': 'text', 0:'count'})]).reset_index(drop=True)

        fig = px.bar(sdf, color='flag', y='count', x=field, barmode="group", hover_data=['text'])

        figures3.add_trace(fig['data'][0], row=k%5 + 1, col=k%3+1)
        figures3.add_trace(fig['data'][1], row=k%5 +1 , col=k%3+1)

    figures3.update_layout(autosize=False, width=1200, height=2000)

    figures.append(dict(data=fig_1, layout=layout_1))
    figures.append(dict(data=fig_2))
    figures.append(dict(data=figures3))

    fdf = pd.DataFrame(columns=['n_mentioned', 'n_not_mentioned'])
    for field in df.drop(columns='review').columns:
        sub_df = df[df[field] != 0]
        fdf.loc[field, 'n_not_mentioned'] = len(sub_df)
        fdf.loc[field, 'n_mentioned'] = len(df) - len(sub_df)
    fdf = fdf.astype(int).reset_index().rename(columns={'index': 'field'})

    fig4 = px.bar(fdf, x = 'field', y = ['n_mentioned', 'n_not_mentioned'], color_discrete_sequence =['green', 'magenta'], width=600)   
    figures.append(dict(data=fig4))

    # plot ids for the html id tag
    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    print('n_images', len(ids))

    return render_template("visualize.html", ids = ids, figuresJSON = figuresJSON)

ml_model = joblib.load(f'{cd}/models/my_model.h5')
vect_tf = joblib.load(f'{cd}/models/viet_vect_fpt.pkl')

categories = ['BUL', 'IT (mạng, nhân viên, hạ tầng dữ liệu, etc)', 'HR', 'PM', 'lead & SA', 'dong_nghiep', 'luong_thuong', 
              'CSVC (canteen, workspace, equipment, etc)', 'tang_ca', 
              'vui_choi_&_nghi_phep', 'thoi_viec', 'bao_hiem', 'ho_tro_# (di_lai, cong_doan, tro_cap#)', 
              'cong_viec', 'dao_tao']
eval_list = ["Không có đánh giá", "Rất ko hài lòng 💩😠", "Chưa hài lòng 😥", 
            "Bình thường 😆", "Hài lòng 🥰", "Rất dzừa lònggg! ❤️💯"]

@app.route('/enhancing', methods=['GET', 'POST'])
def enhanced():
    df = pd.read_csv("data/fpt_reviews_simulation_data.csv").fillna(0)
    cols = df.drop(columns='review').columns.tolist()
    features = [f"r{col}" for col in cols]
    if request.method == "POST":
        reviews = request.form['text']
        data = [request.form[feat] for feat in features]
        data = pd.DataFrame(data = [data], columns=cols, index=[reviews])
        data = data.reset_index().rename(columns={'index': 'review'})
        df = pd.concat([df, data[cols]])
        print(data[cols])
        df.to_csv("data/fpt_reviews_simulation_data.csv", index=False)

    return render_template('enhanced.html')

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