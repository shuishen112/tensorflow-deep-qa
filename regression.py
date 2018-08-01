from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd

import os
import random
from nltk.corpus import stopwords
from nltk import stem
from nltk.tokenize import word_tokenize
import numpy as np
import evaluation
from sklearn import linear_model
from sklearn import svm
import jieba
import nltk
import re
from config import FLAGS
from dataset import QA_dataset
from sklearn.utils import shuffle
import spacy
nlp = spacy.blank("en")

stemmer = stem.lancaster.LancasterStemmer()
english_punctuations = [',','.',':','?','(',')','[',']','!','@','#','%','&']
en_stopwords = stopwords.words('english')

def cut(sentence):

    # doc = nlp(sentence)
    # return [token.text for token in doc if token.text not in en_stopwords]
    words = sentence.lower().split()
    words = [w for w in words if w not in en_stopwords]
    return words
def word_overlap(row):
    question = cut(row["s1"]) 
    answer = cut(row["s2"])
    overlap = set(answer).intersection(set(question)) 
    return len(overlap)         
def idf_word_overlap(row):
    question = cut(row["s1"])
    answer = cut(row["answer"])
    overlap = set(answer).intersection(set(question))
    idf_overlap = 0
    for word in overlap:
        if word in idf_words_dict:
            # small_idf_dict[word] = idf_words_dict[word]
            # if word.isdigit():
                # print word,idf_words_dict[word]
            idf_overlap += idf_words_dict[word]
    # print idf_overlap,len(overlap)
    idf_overlap = (idf_overlap + 1.0) / (len(overlap)+1.0)
    return idf_overlap
def get_features(df):
    df['overlap'] = df.apply(word_overlap,axis = 1)
    # df['idf_overlap'] = df.apply(idf_word_overlap,axis = 1)
    # df['special_feature'] = df.apply(special_Feature,axis = 1)
    names = list()
    names.append("overlap")
    return names
def englishTest():
    data_path = FLAGS.data_path
    train_file = os.path.join(data_path,'train.txt')
    test_file = os.path.join(data_path,'test.txt')
    dev_file = os.path.join(data_path,'test.txt')

    data_set = QA_dataset(train_file,dev_file,test_file,FLAGS)

    train = data_set.train_set
    test = data_set.test_set
    # test = shuffle(test)
    
    # test = test.reindex(np.random.permutation(test.index))
    train = train.reset_index()
    # test = test.reset_index()
    print ('load Data finished')
    columns1 = get_features(train)
    columns2 = get_features(test)
    common = [item for item in columns2 if item in columns1]
    print (common)
    
    x = train[common].fillna(0)
    y = train["flag"]
    test_x = test[common].fillna(0)
    # clf = linear_model.LinearRegression()
    clf = linear_model.LogisticRegression()

    clf.fit(x, y)
    print (clf.coef_)
    # predicted = clf.predict(test_x)
    predicted = clf.predict_proba(test_x)
    predicted = predicted[:,1]
    print (len(predicted))
    print (len(test))
    print (evaluation.evaluationBypandas(test,predicted))
if __name__ == '__main__':
    englishTest()