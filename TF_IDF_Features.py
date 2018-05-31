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
    def process(self,trainHead, trainBody ):
        def merge_head_body(Head):
            res = ' '.join(Head)
            return res

        ## 1. merge text body and head into a list, and transform it into a form easy to vectorize
        head_list = []  # A list of strings, every element is a doc with headline and body.
        for i in range(len(trainHead)):
            head_list.append(merge_head_body(trainHead[i]))
        body_list = []
        for j in range(len(trainBody)):
            body_list.append(merge_head_body(trainBody[j]))
        text_list = head_list + body_list

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

        vec_comb = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        combined_transform = vec_comb.fit_transform(text_list)

        ## 5. Save two pickle for head_transform and Body_transform matrices at current directory
        with open('head_tfidf_transform', "wb") as headfile:
            pickle.dump(head_transform, headfile, -1)
        with open('body_tfidf_transform', "wb") as bodyfile:
            pickle.dump(Body_transform, bodyfile, -1)
        with open('comb_tfidf_transform', "wb") as combfile:
            pickle.dump(combined_transform, combfile, -1)

        '''## 6. Compute the cosine similarity
        simTfidf = pd.Series([ cosine(head_transform.toarray()[i], Body_transform.toarray()[i]) for i in range(len(head_transform.toarray()))])

        return 1-simTfidf'''

        # Extract the stances data
        head_df = pd.read_csv("train_stances.csv")
        body_df = pd.read_csv("train_bodies.csv")

        old_body_IDs = head_df["Body ID"].tolist()
        all_body_IDs = body_df["Body ID"].tolist()
        new_body_IDs = range(len(all_body_IDs))

        # Create a mapping from old body ids to new body ids
        body_id_mapper = dict(zip(all_body_IDs, new_body_IDs))
        new_ID_list = [ body_id_mapper[old_id] for old_id in old_body_IDs ]

        # Compute the cosine similarity
        tfidf_similarities = []
        for head, body in enumerate(new_ID_list):
            head_svd_vector = head_transform[head].toarray()[0]
            body_svd_vector = Body_transform[body].toarray()[0]
            cosine_sim = 1 - cosine(head_svd_vector, body_svd_vector)
            tfidf_similarities.append(cosine_sim)

        return pd.Series(tfidf_similarities)

    def test_process(self, trainHead, trainBody):
        def merge_head_body(Head):
            res = ' '.join(Head)
            return res

        ## 1. merge text body and head into a list, and transform it into a form easy to vectorize
        head_list = []  # A list of strings, every element is a doc with headline and body.
        for i in range(len(trainHead)):
            head_list.append(merge_head_body(trainHead[i]))
        body_list = []
        for j in range(len(trainBody)):
            body_list.append(merge_head_body(trainBody[j]))
        text_list = head_list + body_list
        tfidf_vec = TfidfVectorizer()
        tfidf_vec.fit(text_list)  # Tf-idf calculated on the combined training + test set
        vocabulary = tfidf_vec.vocabulary_
        with open('test_vocabulary_tfidf',"wb") as vocabfile:  ## 5.31 test vocaulary should be different
            pickle.dump(vocabulary, vocabfile, -1)
        vecH = TfidfVectorizer(vocabulary=vocabulary)
        head_str_list = [' '.join(h) for h in trainHead]
        head_transform = vecH.fit_transform(
            head_str_list)  # use ' '.join(Headline_unigram) instead of Headline since the former is already stemmed
        # obtain a sparse matrix

        ## 4. Fit the body into the same vocabulary
        vecB = TfidfVectorizer(vocabulary=vocabulary)
        Body_str_list = [' '.join(k) for k in trainBody]
        Body_transform = vecB.fit_transform(Body_str_list)

        ## 5. Save two pickle for head_transform and Body_transform matrices at current directory
        with open('test_head_tfidf_transform', "wb") as headfile:
            pickle.dump(head_transform, headfile, -1)
        with open('test_body_tfidf_transform', "wb") as bodyfile:
            pickle.dump(Body_transform, bodyfile, -1)
        vec_comb = TfidfVectorizer(vocabulary=vocabulary)
        combined_transform = vec_comb.fit_transform(text_list)
        with open('test_comb_tfidf_transform', "wb") as combfile:
            pickle.dump(combined_transform, combfile, -1)

        head_df = pd.read_csv("competition_test_stances.csv")
        body_df = pd.read_csv("competition_test_bodies.csv")

        old_body_IDs = head_df["Body ID"].tolist()
        all_body_IDs = body_df["Body ID"].tolist()
        new_body_IDs = range(len(all_body_IDs))

        # Create a mapping from old body ids to new body ids
        body_id_mapper = dict(zip(all_body_IDs, new_body_IDs))
        tfidf_similarities = []
        new_ID_list = [body_id_mapper[old_id] for old_id in old_body_IDs]
        for head, body in enumerate(new_ID_list):
            head_svd_vector = head_transform[head].toarray()[0]
            body_svd_vector = Body_transform[body].toarray()[0]
            cosine_sim = 1 - cosine(head_svd_vector, body_svd_vector)
            tfidf_similarities.append(cosine_sim)

        tfidf_sim = pd.Series(tfidf_similarities)
        with open('test_tfidf_sim', "wb") as bodyfile:
            pickle.dump(tfidf_sim, bodyfile, -1)

        with open('test_body_id_reference', "wb") as file:
            pickle.dump(new_ID_list, file, -1)
