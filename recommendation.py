import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
import random
import nltk

hotel = pd.read_csv("nusatripfix.csv", header=0)
#preprocessing


def clean_lower(lwr):
    lwr = lwr.lower() # lowercase text
    return lwr

# Buat kolom tambahan untuk data description yang telah dicasefolding  
hotel['lwr'] = hotel['deskripsi'].apply(clean_lower)
#Remove Puncutuation
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z]')


def remove_punct(text):
    text = clean_spcl.sub('', text)
    text = clean_symbol.sub(' ', text)
    return text

# Buat kolom tambahan untuk data description yang telah diremovepunctuation   
hotel['remove_punct'] = hotel['lwr'].apply(remove_punct)
def _normalize_whitespace(text):
    """
    This function normalizes whitespaces, removing duplicates.
    """
    corrected = str(text)
    corrected = re.sub(r"//t",r"\t", corrected)
    corrected = re.sub(r"( )\1+",r"\1", corrected)
    corrected = re.sub(r"(\n)\1+",r"\1", corrected)
    corrected = re.sub(r"(\r)\1+",r"\1", corrected)
    corrected = re.sub(r"(\t)\1+",r"\1", corrected)
    return corrected.strip(" ")
hotel['remove_double_ws'] = hotel['remove_punct'].apply(_normalize_whitespace)
#clean stopwords
stw = open("sw.txt")
# Use this to read file content as a stream:
line = stw.read()
stopword = line.split()

def clean_stopwords(text):
    text = ' '.join(word for word in text.split() if word not in stopword) # hapus stopword dari kolom deskripsi
    return text

# Buat kolom tambahan untuk data description yang telah distopwordsremoval   
hotel['remove_sw'] = hotel['remove_double_ws'].apply(clean_stopwords)
#tambahan stopword
tambahan = pd.DataFrame(hotel['remove_sw'])
hotel['tambah_swr']= tambahan.replace(to_replace =['also','always','alse',
                                        'another','make','please','may',
                                       'take','want'],  
                            value ="", regex= True) 
nltk.download('wordnet')
wn= nltk.WordNetLemmatizer()
def lemmatization(text):
    text = ' '.join(wn.lemmatize(word) for word in text.split() if word in text)
    return text

# Buat kolom tambahan untuk data description yang telah dilemmatization   
hotel['desc_remove_lemma'] = hotel['tambah_swr'].apply(lemmatization)
hotel['desc_removefix']=hotel['desc_remove_lemma']
# Deskripsi baru (bersih)
def print_description_clean(index):
    example = hotel[hotel.index == index][['desc_removefix', 'namahotel']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Nama Hotel:', example[1])

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer= CountVectorizer(encoding='latin-1', ngram_range=(1,1), 
                                  tokenizer=None, analyzer='word',
                                  stop_words= None)
countvec= count_vectorizer.fit_transform(hotel['desc_removefix']).toarray()
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


