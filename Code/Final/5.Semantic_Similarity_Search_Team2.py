
import torch
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import scipy.spatial
from transformers import BertTokenizer, BertModel
import os
import sys
sys.path.insert(0, os.getcwd())
from Utils_Team2 import *  # Call functions as Utils
#%%
###############################################################################
# 1. Recompose Job Corpus dataset
###############################################################################
#recommed a job title from job discription based on similarty with resume 
# Load your job descriptions dataset
url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation.csv'
df = pd.read_csv(url)

# Combining text columns into a single column (Can recompose corpus of individual categories)
text_columns = ['Description', 'Description_Abilities',
                'Description_Knowledge', 'Description_Skills', 'Description_Tech', 'Description_Interests']
df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1) #'Description_Job'

# Handle missing values
df['Title'].fillna('NA', inplace=True)
df['combined_text'].fillna('NA', inplace=True)

# Creating a new DataFrame with only the relevant columns
job_df = df[['Title', 'combined_text']]

###############################################################################
# 2. Get Embedding vectors of each occupation
# Semantic Search using Siamese-BERT Networks (Sentence-BERT)
# The produced embedding vector will be used to match job - resume using cosine similarity.
###############################################################################

def encode_text(text):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(text)
    return sentence_embeddings

# Preprocess and save job encodings
job_encodings = {}
for index, row in job_df.iterrows():
    job_title = row['Title']
    job_description = row['combined_text']
    job_encoding = encode_text(job_description)
    job_encodings[job_title] = job_encoding

#%%
# Save the encodings to a file
with open('/home/ubuntu/Project/job_encodings.pkl', 'wb') as f:
    pickle.dump(job_encodings, f)

# # Load the precomputed job encodings
# with open('Job_corpus_embeddings.npy', 'rb') as f:
#     job_encodings = pickle.load(f)

