import gensim
import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import cosine
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dense import removeStop, DensitySearch

class Sentiment_Feature:
    def __init__(self,stopword, pieces = 3):
        self.stopword = stopword
        self.pieces = pieces

    ## Do a sentiment analysis for each headline, select paragraoh, and whole text body and return the compound mark for each one
    #  input: trainHead inputs are n-gram processed 2D arrays of stemmed tokens. A[i] = represent a token list of a certain file.
    #         trainBody_sentences is a 2D array where each element is a preprocessed sentence. ( helper.body_to_sentences)
    #  output: return 3 panda series, each of them represent compound for either headline or body text.
    def sentiment_proc(self, trainHead, trainBody_sentences, body_ID_head):
        sia = SentimentIntensityAnalyzer()

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

        headline_mark = []
        headline_keyword = []
        counter = 0
        headline_tfidf_vec = head_tfidf.toarray()
        for headline in trainHead:
            temp =[]
            for word in headline:
                index = vocabulary[word]
                temp.append((headline_tfidf_vec[counter][index],word))
            temp.sort(reverse=True)
            polarity = sia.polarity_scores(' '.join(headline))
            headline_mark.append(polarity['compound'])
            headline_keyword.append(temp[:3])

        body_mark_ave = []
        for doc in trainBody_sentences:
            body_mark = []
            for sentence in doc:
                polarity2 = sia.polarity_scores(sentence)
                body_mark.append(polarity2['compound'])
            body_mark_ave.append(np.average(body_mark))

        head_piece_mark_ave=[]
        for k in len(trainHead):
            temp_word = []
            temp_weight = []
            head_piece_mark = []
            for val,word in headline_keyword[k]:
                temp_word.append(word)
                temp_weight.append(val)
            dense_piece = DensitySearch(temp_word, '.'.join(trainBody_sentences[body_ID_head[k]]), self.pieces, temp_weight)
            if dense_piece:
                for sentence2 in dense_piece:
                    polarity3 = sia.polarity_scores(sentence2)
                    head_piece_mark.append(polarity3['compound'])
                head_piece_mark_ave.append(np.average(head_piece_mark))
            else:
                head_piece_mark_ave.append(-2.0)

        return pd.Series(headline_mark), pd.Series(body_mark_ave), pd.Series(head_piece_mark_ave)



