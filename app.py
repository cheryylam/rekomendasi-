
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')

hotel = pd.read_csv("lemmatization.csv", header=0)
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer= CountVectorizer(encoding='latin-1', ngram_range=(1,1), 
                                  tokenizer=None, analyzer='word',
                                  stop_words= None)
countvec= count_vectorizer.fit_transform(hotel['deskripsi']).toarray()
#TF IDF
from sklearn.feature_extraction.text import TfidfTransformer
transformer= TfidfTransformer(norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False)
tfidf= transformer.fit_transform(countvec)  
#cosine
cos_sim= cosine_similarity(tfidf, tfidf)
#recommendation
# Set index utama di kolom 'namahotel'
hotel.set_index('namahotel', inplace=True)
indices = pd.Series(hotel.index)
my_array=indices.to_numpy()
cheryls=my_array

def recommendations(namahotel, cos_sim = cos_sim):
    
    recommended_hotel = []
    
    # Mengambil nama hotel berdasarkan variabel indicies
    idx = indices[indices == namahotel].index[0]

    # Membuat series berdasarkan skor kesamaan
    score_series = pd.Series(cos_sim[idx]).sort_values(ascending = False)

    # mengambil index dan dibuat 10 baris rekomendasi terbaik
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    for i in top_10_indexes:
        recommended_hotel.append(list(hotel.index)[i])
    print(recommended_hotel)
    return recommended_hotel


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('home.html', cheryls=cheryls)

    if request.method == 'POST':
        hotels = request.form['daftarhotel']
        res = recommendations(hotels)
        names=[]
        for i in range(len(res)):
            names.append(res[i])
        return render_template('akhir.html', result=names, cheryls=cheryls)

    else:
        return render_template('home.html')
 

if __name__ == '__main__':
    app.run(debug=True)
