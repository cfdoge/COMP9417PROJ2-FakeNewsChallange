import gensim
import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize

class Word2Vec_Feature():
    def __init__(self, model_name='default'):
        if model_name == 'default':
            self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        else:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True)

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

    def topk_tfidf(self,body, tfidf, vocabulary):
        vec = []
        # print(tfidf)
        bag = set(body)
        for word in bag:
            # print(word)
            try:
                index = vocabulary[word.lower()]
                # print('wtf')
                vec.append((tfidf.toarray()[0][index], word))
            except:
                # print('w')

                vec.append((0, word))
        vec.sort(reverse=True)
        output = []
        return vec[:10]

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
        head_tfidf_array = head_tfidf
        for i in range(len(trainHead)):
            head_mat = np.zeros(300, )
            tf_idf_vec = head_tfidf_array[i].toarray()[0]
            for word in trainHead[i]:
                try:
                    index = vocabulary[word.lower()]
                    head_mat += model[word]*tf_idf_vec[index]
                except:
                    continue
            head_w2v.append(head_mat)

            # head_w2v.append([model[word] for word in trainHead[i] if word in model])

        body_w2v = []
        body_tfidf_array = body_tfidf
        for j in range(len(trainBody)):
            body_mat = np.zeros(300, )
            tf_idf_vec2 = body_tfidf_array[j]
            topk_vec = self.topk_tfidf(trainBody[j], body_tfidf_array[j], vocabulary)
            # print(topk_vec)
            for val2, word2 in topk_vec:
                try:
                    body_mat += model[word2] * val2
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

    def test_weighted_transform(self, trainHead, trainBody):
        model = self.model
        ## Read all tf-idf related pickle files
        filename_hvec = 'test_head_tfidf_transform'
        with open(filename_hvec, "rb") as infile1:
            head_tfidf = pickle.load(infile1)

        filename_bvec = 'test_body_tfidf_transform'
        with open(filename_bvec, "rb") as infile2:
            body_tfidf = pickle.load(infile2)

        filename_vocab = 'test_vocabulary_tfidf'
        with open(filename_vocab, "rb") as infile3:
            vocabulary = pickle.load(infile3)

        ## Calculate the weighted w2v vector
        head_w2v = []
        head_tfidf_array = head_tfidf
        for i in range(len(trainHead)):
            head_mat = np.zeros(300, )
            tf_idf_vec = head_tfidf_array[i].toarray()[0]
            for word in trainHead[i]:
                try:
                    index = vocabulary[word.lower()]
                    head_mat += model[word] * tf_idf_vec[index]
                except:
                    continue
            head_w2v.append(head_mat)

            # head_w2v.append([model[word] for word in trainHead[i] if word in model])

        body_w2v = []
        body_tfidf_array = body_tfidf
        for j in range(len(trainBody)):
            body_mat = np.zeros(300, )
            tf_idf_vec2 = body_tfidf_array[j]
            topk_vec = self.topk_tfidf(trainBody[j], body_tfidf_array[j], vocabulary)
            # print(topk_vec)
            for val2, word2 in topk_vec:
                try:
                    body_mat += model[word2] * val2
                except:
                    continue
            body_w2v.append(body_mat)

        # head_tfidf_norm = np.array(normalize([head_tfidf]))[0]
        # body_tfidf_norm = np.array(normalize([body_tfidf]))[0]

        # head_w2v_weighted = np.multiply(head_w2v, head_tfidf_norm)
        # body_w2v_weighted = np.multiply(body_w2v, body_tfidf_norm)

        with open('test_head_w2v_tfidf_transform', "wb") as headfile:
            pickle.dump(head_w2v, headfile, -1)
        with open('test_body_w2v_tfidf_transform', "wb") as bodyfile:
            pickle.dump(body_w2v, bodyfile, -1)

        return head_w2v, body_w2v



    ## calculate the cosine similarity of head and body vectors
    # input:  head_w2v: a list of headlines represented in w2v form.
    #         body_w2v: a list of body text represented in w2v form.
    # output: A Series of cosine similarity for each pair
    def cosin_similarity(self,head_w2v, body_w2v):
        with open('body_id_reference', 'rb') as infile:
            ref_id = pickle.load(infile)
        # print(ref_id)
        simTfidf = pd.Series([cosine(head_w2v[i], body_w2v[ref_id[i]]) for i in range(len(ref_id))])
        with open('w2v_sim', 'wb') as file:
            pickle.dump(1-simTfidf, file, -1)

        return 1 - simTfidf

    def test_cosin_similarity(head_w2v, body_w2v):
        with open('test_body_id_reference', 'rb') as infile:
            ref_id = pickle.load(infile)
        # print(ref_id)
        simTfidf = pd.Series([cosine(head_w2v[i], body_w2v[ref_id[i]]) for i in range(len(ref_id))])
        with open('test_w2v_sim', 'wb') as file:
            pickle.dump(1-simTfidf, file, -1)
        return 1 - simTfidf


