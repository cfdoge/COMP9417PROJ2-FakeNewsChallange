import numpy as np
import sklearn as sk
import nltk
import pandas as pd
import gensim

# read data



## ----------------
## input:
##      ngram: 1,2 or 3 ngram processing
##      stopwords: True or False
##      match_mod: ratio
## output:
##      title_body_match_rate: a np array which each row which represent that
##                              how many grams are similar in bodies and title.
##
##
class CounterFeature():
    def __init__(self, ngram, stopwords, match_mod):
        ngram = ngram
        self.stopwords = stopwords
        self.match_mod = match_mod
        train_body_df = pd.read_csv('train_bodies.csv')
        train_stance_df = pd.read_csv('train_stances.csv')
        self.data_df = pd.merge(train_body_df, train_stance_df, how = 'right', on='Body ID')

    def n_gram_process(self,body,title):

        title_gram = []
        body_gram = []
        distinct_title_gram = 0
        distinct_body_gram = 0
        matched_number = 0
        # use of gram function
        for token in title_gram:
            if token in body_gram:
                matched_number += 1
        matched_gram = []


        return matched_number, matched_gram, distinct_title_gram

    def match_process(self, matched_number, distinct_title_gram):
        if self.match_mod == 'ratio':
            match_ratio = self.matched_number / self.distinct_title_gram
        return match_ratio

    def output(self):
        ratio =[]
        for i in  range(self.data_df.shape[0]):
            matched_number, matched_gram, distinct_title_gram =self.n_gram_process(self.data_df[i]['articleBody'], self.data_df[i]['Headline'])
            ratio.append(self.match_process(matched_number, matched_gram, distinct_title_gram))
        counters = pd.Series(ratio)
        return counters


