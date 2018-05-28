import numpy as np
import pandas as pd
import pickle as pkl
from scipy import linalg
from scipy.spatial.distance import cosine

class Deserializer:
    def __init__(self, headlines, bodies):
        ''' Deserialize the TF-IDF data '''
        self.head_collection = self.deserialize_data(headlines)
        self.body_collection = self.deserialize_data(bodies)
        self.data = self.merge_data()

    def deserialize_data(self, filename):
        with open(filename, "rb") as f:
            data = pkl.load(f)
        return data.toarray()

    def merge_data(self):
        head_toMatrix = np.matrix(self.head_collection)
        body_toMatrix = np.matrix(self.body_collection)
        return np.concatenate((head_toMatrix, body_toMatrix))

class LSA:
    def __init__(self):
        self.inverse_sigma = None
        self.transpose_U = None

    def decomposition(self, term_document_matrix, num_topics):
        ''' Given a term-document matrix represented in a vector feature-space,
            decompose the matrix into a new latent-topic feature space '''

        # DEBUG: print out the original term_document_matrix for comparison
        # print(term_document_matrix, end="\n\n")

        # decompose the term document matrix
        U, Sigma, Vh = linalg.svd(term_document_matrix)
        # extract dimensions
        num_terms, num_docs = term_document_matrix.shape
        # compute the diagonal singular value matrix
        SigmaDiag = linalg.diagsvd(Sigma, num_terms, num_docs)

        # DEBUG: confirm that the matrix product of U•Sigma•Vh results in a
        #        close approximation of the original term_document_matrix.
        # matrix_recomposition = U.dot(SigmaDiag.dot(Vh))
        # print(matrix_recomposition, end="\n\n")

        # num_topics must be <= num_docs
        if num_topics > num_docs:
            return False
        # trim excess latent topics
        new_U = U[:,:num_topics]    # trims term-to-topic matrix
        new_Vh = Vh[:num_topics,:]  # trims document-to-topic matrix
        new_SigmaDiag = SigmaDiag[:num_topics,:num_topics] # trims to a square matrix

        # DEBUG: confirm that the matrix product of new_U•Sigma•Vh results in a
        #        decent approximation of the original term_document_matrix, but not
        #        as accurate as U•Sigma•Vh, as there are fewer latent topics.
        # new_matrix_recomposition = new_U.dot(new_SigmaDiag.dot(new_Vh))
        # print(new_matrix_recomposition, end="\n\n")

        # store the matrix components necessary to convert a vector (e.g. TF-IDF)
        # in the original feature space to a vector in the latent topic feature space:
        #    • the inverse of the diagonal singular value matrix, new_Sigma; and
        #    • the transpose of the term-to-topic matrix, new_U
        self.inverse_sigma = linalg.inv(new_SigmaDiag)
        self.transpose_U = new_U.transpose()

    def transform(self, documents):
        ''' Transforms all documents in the old feature space into LSA feature space '''
        return [ self.project_onto_latent_featurespace(document) for document in documents ]

    def project_onto_latent_featurespace(self, document):
        ''' Given a document represented in the old vector format (eg. TF-IDF),
            derive its vector format in the new latent-topic feture space. The
            project formula is defined as:

                    d-hat = inverse(Sig) • transpose(U) • d

            where d-hat is the new vector and d is the old vector '''
        new_document = self.inverse_sigma.dot(self.transpose_U.dot(document))
        return new_document


if __name__ == "__main__":

    ######## EXAMPLE USAGE ########

    # 1. initialise deserializer to extract the TF-IDF feature data
    deserializer = Deserializer(headlines="head_tfidf_transform", bodies="body_tfidf_transform")

    # 2. transpose the document-term matrix to get the term-document matrix
    term_document_matrix = deserializer.data.transpose()

    # 3. instantiate the LSA object and decompose the term-document matrix
    lsa = LSA()
    lsa.decomposition(term_document_matrix, num_topics=3)

    # 4. transform the headline and body texts separately
    new_head = lsa.transform(deserializer.head_collection)
    new_body = lsa.transform(deserializer.body_collection)

    # 5. compute the cosine similarity,  (1 - distance), for each headline-body pair
    simSVD = pd.Series([ (1 - cosine(new_head[i], new_body[i])) for i in range(len(new_head)) ])

    # 6. write the SVD similarity data out to file
    with open("svd_similarity.pkl", "wb") as f:
        pkl.dump(simSVD, f, -1)
