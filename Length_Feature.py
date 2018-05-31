import pandas as pd
import pickle as pkl

class LengthFeatures:

    def __init__(self, mode="train",
                       stance_file="train_stances.csv",
                       body_file="train_bodies.csv",
                       output_file="train_length_features"):

        self.body_file = body_file       # title of body file
        self.stance_file = stance_file   # title of head file
        self.output_file = output_file   # title of output file

    def generate_length_features(self):
        # 1. Read files
        body_df = pd.read_csv(self.body_file )
        head_df = pd.read_csv(self.stance_file )

        # 2. Create mapping from Body ID to Body text index
        old_body_IDs = head_df["Body ID"].tolist()
        all_body_IDs = body_df["Body ID"].tolist()
        new_body_IDs = range(len(all_body_IDs))

        body_id_mapper = dict(zip(all_body_IDs, new_body_IDs))
        new_ID_list = [ body_id_mapper[old_id] for old_id in old_body_IDs ]

        # 3. Extract the lengths for each body
        body_texts = body_df["articleBody"].tolist()
        body_lengths = [ len(body.split()) for body in body_texts ]

        # 4. Join body length to headlines on body ID
        lengths = [ body_lengths[body] for body in new_ID_list ]

        # 5. Generate pickle file
        with open(self.output_file, "wb") as f:
            pkl.dump(lengths, f, -1)

'''if __name__ == "__main__":

    lf = LengthFeatures() # NB: default setting is training
    lf.generate_length_features()
'''
