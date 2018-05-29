import gensim
import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize

class Word2Vec_Feature():
    def __init__(self, model_name='default'):
        if model_name == 'default':
            self.model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        else:
            self.model = gensim.models.Word2Vec.load_word2vec_format(model_name, binary=True)

    ## vectorize and transform to w2v format
    #  input: Both inputs are n-gram processed 2D arrays of stemmed tokens. A[i] = represent a token list of a certain file.
    #  output: return 2 list, the first one is headlines vector in w2v form and the second is body vector.
    def transform(self, trainHead, trainBody):
        model = self.model
        head_w2v =[]
        for i in range(len(trainHead)):
            head_mat = np.zeros(300, )
            for word in trainHead[i]:
                try:
                    head_mat += model[word]
                except:
                    continue
            head_w2v.append(head_mat)

            #head_w2v.append([model[word] for word in trainHead[i] if word in model])

        body_w2v = []
        for j in range(len(trainBody)):
            body_mat = np.zeros(300,)
            for word2 in trainBody[j]:
                try:
                    body_mat += model[word2]
                except:
                    continue
            body_w2v.append(body_mat)
            #body_w2v.append([model[word] for word in trainBody[j] if word in model])

        with open('head_w2v_transform', "wb") as headfile:
            pickle.dump(head_w2v, headfile, -1)
        with open('body_w2v_transform', "wb") as bodyfile:
            pickle.dump(body_w2v, bodyfile, -1)

        return head_w2v,body_w2v

    ## vectorize and transform to w2v format, weighted by tf_idf value
    #  input: Both inputs are n-gram processed 2D arrays of stemmed tokens. A[i] = represent a token list of a certain file.
    #  output: return 2 list, the first one is headlines vector in w2v form and the second is body vector in w2v form
    #  Both vectors now are weighted by tf_idf value.
    def weighted_transform(self, trainHead, trainBody):
        model = self.model

        ## Read all tf-idf related pickle files
        filename_hvec = 'head_tfidf_transform'
        with open(filename_hvec, "rb") as infile1:
            head_tfidf = pickle.load(infile1)

        filename_bvec = 'body_tfidf_transform'
        with open(filename_bvec, "rb") as infile2:
            body_tfidf = pickle.load(infile2)

        filename_vocab = 'vocabulary_tfidf'
        with open(filename_vocab, "rb") as infile3:
            vocabulary = pickle.load(infile3)

        ## Calculate the weighted w2v vector
        head_w2v = []
        head_tfidf_array = head_tfidf.toarray()
        for i in range(len(trainHead)):
            head_mat = np.zeros(300, )
            tf_idf_vec = head_tfidf_array[i]
            for word in trainHead[i]:
                try:
                    index = vocabulary[word]
                    head_mat += model[word]*tf_idf_vec[index]
                except:
                    continue
            head_w2v.append(head_mat)

            # head_w2v.append([model[word] for word in trainHead[i] if word in model])

        body_w2v = []
        body_tfidf_array = body_tfidf.toarray()
        for j in range(len(trainBody)):
            body_mat = np.zeros(300, )
            tf_idf_vec2 = body_tfidf_array[j]
            for word2 in trainBody[j]:
                try:
                    index2 = vocabulary[word2]
                    body_mat += model[word2]*tf_idf_vec2[index2]
                except:
                    continue
            body_w2v.append(body_mat)

        #head_tfidf_norm = np.array(normalize([head_tfidf]))[0]
        #body_tfidf_norm = np.array(normalize([body_tfidf]))[0]



        #head_w2v_weighted = np.multiply(head_w2v, head_tfidf_norm)
        #body_w2v_weighted = np.multiply(body_w2v, body_tfidf_norm)

        with open('head_w2v_tfidf_transform', "wb") as headfile:
            pickle.dump(head_w2v, headfile, -1)
        with open('body_w2v_tfidf_transform', "wb") as bodyfile:
            pickle.dump(body_w2v, bodyfile, -1)

        return head_w2v,body_w2v

    ## calculate the cosine similarity of head and body vectors
    # input:  head_w2v: a list of headlines represented in w2v form.
    #         body_w2v: a list of body text represented in w2v form.
    # output: A Series of cosine similarity for each pair
    def cosin_similarity(self, head_w2v, body_w2v):
        simTfidf = pd.Series([cosine(head_w2v[i], body_w2v[i]) for i in range(len(head_w2v))])
        return simTfidf

