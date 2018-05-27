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
        head_w2v = []
        for i in range(len(trainHead)):
            head_w2v.append([model[word] for word in trainHead[i] if word in model])

        body_w2v = []
        for j in range(len(trainBody)):
            body_w2v.append([model[word] for word in trainBody[j] if word in model])

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
        head_w2v = []
        for i in range(len(trainHead)):
            head_w2v.append([model[word] for word in trainHead[i] if word in model])

        body_w2v = []
        for j in range(len(trainBody)):
            body_w2v.append([model[word] for word in trainBody[j] if word in model])

        filename_hvec = 'head_tfidf_transform'
        with open(filename_hvec, "rb") as infile:
            head_tfidf = pickle.load(infile)

        filename_bvec = 'body_tfidf_transform'
        with open(filename_bvec, "rb") as infile:
            body_tfidf = pickle.load(infile)

        head_tfidf_norm = np.array(normalize([head_tfidf]))[0]
        body_tfidf_norm = np.array(normalize([body_tfidf]))[0]

        head_w2v_weighted = np.multiply(head_w2v, head_tfidf_norm)
        body_w2v_weighted = np.multiply(body_w2v, body_tfidf_norm)

        with open('head_w2v_tfidf_transform', "wb") as headfile:
            pickle.dump(head_w2v_weighted, headfile, -1)
        with open('body_w2v_tfidf_transform', "wb") as bodyfile:
            pickle.dump(body_w2v_weighted, bodyfile, -1)

        return head_w2v_weighted,body_w2v_weighted

    ## calculate the cosine similarity of head and body vectors
    # input:  head_w2v: a list of headlines represented in w2v form.
    #         body_w2v: a list of body text represented in w2v form.
    # output: A Series of cosine similarity for each pair
    def cosin_similarity(self, head_w2v, body_w2v):
        simTfidf = pd.Series([cosine(head_w2v[i], body_w2v[i]) for i in range(len(head_w2v))])
        return simTfidf

