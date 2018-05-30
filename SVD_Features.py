from sklearn.decomposition import TruncatedSVD
from scipy import linalg
import numpy as np
from scipy.spatial.distance import cosine
import pickle as pkl
import pandas as pd


# 1. Extract combined TF-IDFs of headlines and bodies
with open("comb_tfidf_transform", "rb") as f:
    heads_and_bodies_tfidf = pkl.load(f)
with open("head_tfidf_transform", "rb") as f:
    head_tfidf_vectors = pkl.load(f)
with open("body_tfidf_transform", "rb") as f:
    body_tfidf_vectors = pkl.load(f)

# 2. Extract the stances data
head_df = pd.read_csv("train_stances.csv")
body_df = pd.read_csv("train_bodies.csv")

old_body_IDs = head_df["Body ID"].tolist()
all_body_IDs = body_df["Body ID"].tolist()
new_body_IDs = range(len(all_body_IDs))

# 3. Create a mapping from old body ids to new body ids
body_id_mapper = dict(zip(all_body_IDs, new_body_IDs))
new_ID_list = [ body_id_mapper[old_id] for old_id in old_body_IDs ]

# 4. Fit the TF-IDF matrix to the SVD model
n_latent_topics = 50

svd = TruncatedSVD(n_components=n_latent_topics, n_iter=15)
svd.fit(heads_and_bodies_tfidf)

# 5. Extract the U and Sigma-inverse from the SVD
U = svd.components_
Sigma = linalg.diagsvd(svd.singular_values_, n_latent_topics, n_latent_topics)

# 6. Transform the TF-IDF vectors for corresponding headline and body texts
#    into the new (latent) feature space
svd_transform = svd.transform(heads_and_bodies_tfidf)

# 7. Separate svd_transform into transformed headline and boy texts
svd_headlines = svd_transform[:49972]
svd_bodies = svd_transform[49972:]

# 8. Calculate Cosine similarity between headlines and their corresponding body texts
svd_similarities = []
for head, body in enumerate(new_ID_list):
    head_svd_vector = svd_headlines[head]
    body_svd_vector = svd_bodies[body]
    cosine_sim = (1 - cosine(head_svd_vector, body_svd_vector))
    svd_similarities.append(cosine_sim)

# 9. Write the SVD cosine similarity list to file as a pd.Series
'''with open("svd_cosine_similarity", "wb") as f:
    pkl.dump(pd.Series(svd_similarities), f, -1)'''
