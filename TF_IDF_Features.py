import numpy as np
import sklearn as sk
import nltk
import pandas as pd
import gensim
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

class Tf_Idf_Fea:
    def __init__(self):
        pass

    ## vectorize and processing training text
    #  input: Both inputs are n-gram processed 2D arrays of stemmed tokens. A[i] = represent a token list of a certain file.
    #  output: cosine similarity in pandas Series form
    def process(trainHead, trainBody ):
        def merge_head_body(Head, Body):
            res = '%s %s' % (' '.join(Head), ' '.join(Body))
            return res
        ## 1. merge text body and head into a list, and transform it into a form easy to vectorize
        text_list = [] # A list of strings, every element is a doc with headline and body.
        for i in range(len(trainHead)):
            text_list.append(merge_head_body(trainHead[i], trainBody[i]))

        ## 2. Using tfidf Vectorizer to get the vocabulary for the whole training corpus
        ## Trian the vocabulary first.
        tfidf_vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2)
        tfidf_vec.fit(text_list)  # Tf-idf calculated on the combined training + test set
        vocabulary = tfidf_vec.vocabulary_
        with open('vocabulary_tfidf', "wb") as vocabfile: ## 5.29 Updated: We need the vocabulary for the whole transformation latter
            pickle.dump(vocabulary, vocabfile, -1)
        ## 3.Fit the headline into the same vocabulary
        vecH = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        head_str_list = [ ' '.join(h) for h in trainHead]
        head_transform = vecH.fit_transform(head_str_list)  # use ' '.join(Headline_unigram) instead of Headline since the former is already stemmed
        # obtain a sparse matrix

        ## 4. Fit the body into the same vocabulary
        vecB = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        Body_str_list = [' '.join(h) for h in trainBody]
        Body_transform = vecB.fit_transform(Body_str_list)

        ## 5. Save two pickle for head_transform and Body_transform matrices at current directory
        with open('head_tfidf_transform', "wb") as headfile:
            pickle.dump(head_transform, headfile, -1)
        with open('body_tfidf_transform', "wb") as bodyfile:
            pickle.dump(Body_transform, bodyfile, -1)

        ## 6. Compute the cosine similarity
        simTfidf = pd.Series([ cosine(head_transform.toarray()[i], Body_transform.toarray()[i]) for i in range(len(head_transform.toarray()))])

        return 1-simTfidf
