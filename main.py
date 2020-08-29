# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:27:23 2020

@author: COM
"""

import pandas as pd
import numpy as np
import pickle
import operator
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics.pairwise import cosine_similarity
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
from flask import Flask,request,jsonify,render_template,url_for,send_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

app=Flask(__name__)
#model=pickle.load(open(r"D:/BakejobsCourses/codes/deploy/banking-faq-bot-master/modelqa.pickle",'rb'))


model=pickle.load(open("modelqa.pickle",'rb'))

#vector=pickle.load(open(r"D:/BakejobsCourses/codes/deploy/banking-faq-bot-master/vector.pickle",'rb'))

#vector = joblib.load("D:/BakejobsCourses/codes/deploy/banking-faq-bot-master/vector.pickle")



vector = joblib.load("vector.pickle")



#data = pd.read_csv('D:/BakejobsCourses/codes/deploy/banking-faq-bot-master/BankFAQs.csv')

data = pd.read_csv('BankFAQs.csv')
questions = data['Question'].values

le = LE()

tfv = TfidfVectorizer(min_df=1, stop_words='english')


def get_max5(arr):
    ixarr = []
    for ix, el in enumerate(arr):
        ixarr.append((el, ix))
    ixarr.sort()

    ixs = []
    for i in ixarr[-5:]:
        ixs.append(i[1])

    return ixs[::-1]


stemmer = LancasterStemmer()
def cleanup(sentence):
    word_tok = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(w) for w in word_tok]

    return ' '.join(stemmed_words)



@app.route('/api',methods=["GET"])
def chat():
    try:
        print("Bot: Hi, Welcome to our bank!")
        #usr=request.args.get('usr')
        usr=request.args.get('usr')
        class_=request.args.get('class')
        #usr = request.form['usr']
        #usr="For what period can I open a Recurring Deposit"
        usr=''.join(usr)
        usr=usr.strip().lower()
        print(usr)
        

        
        #tfv = TfidfVectorizer(min_df=1, stop_words='english')
        tfv = TfidfVectorizer()
        print([cleanup(usr.strip().lower())])
        
        #tfv.fit([cleanup(usr.strip().lower())])
        #t_usr = tfv.transform([cleanup(usr.strip().lower())])
        
        #instantiate CountVectorizer() 
        
        #count_vector=TfidfVectorizer.transform([cleanup(usr.strip().lower())]) 
        #print(count_vector)
        
        t_usr = vector.transform([cleanup(usr.strip().lower())])
        
        print(t_usr[0])
        
        #class_ = le.inverse_transform(model.predict(t_usr[0]))
        #class_ ='cards'
        print(class_)
        newclass=''.join(class_)
        questionset = data[data['Class']==newclass]
        
        print(questionset)
        
#        t_usr = tfv.transform([cleanup(usr.strip().lower())])
#        class_ = le.inverse_transform(model.predict(t_usr)[0])
#        questionset = data[data['Class']==class_]
        
        
        cos_sims = []
        for question in questionset['Question']:
            #sims = cosine_similarity(tfv.transform([question]), t_usr)
            sims = cosine_similarity(vector.transform([question]), t_usr)
            cos_sims.append(sims)
            
        ind = cos_sims.index(max(cos_sims))
        question = questionset["Question"][questionset.index[ind]]
        print("Bot:", data['Answer'][ind])
        
        inds=get_max5(cos_sims)
        q_cnt = 1
        moreq=[]
        for ix in inds:
            print(q_cnt,"Question: "+data['Question'][questionset.index[ix]])
            # print("Answer: "+data['Answer'][questionset.index[ix]])
            print('-'*50)
            q_cnt += 1
            moreq.append(data['Question'][questionset.index[ix]])
            
        print(data['Answer'][ind])
        
        return jsonify(data['Answer'][ind], inds, moreq)
        #return '123'
    except Exception as e:
        return str(e)
if __name__=='__main__':
     app.run(debug=True)   
     
     
     
#!pip install pipreqs
#
#!pipreqs D:/BakejobsCourses/codes/deploy/banking-faq-bot-master
