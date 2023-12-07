#%%
###############################################################################
# Environment: Cloud setting (will be cleaned or substituted when submit
###############################################################################
### Project folder on the Cloud
import os
# Step 1: Check current path
current_path = os.getcwd()
print(f"Current working directory: {current_path}")
# Step 2: Create /home/ubuntu if it doesn't exist
target_directory = '/home/ubuntu'
if not os.path.exists(target_directory):
    try:
        os.makedirs(target_directory)
        print(f"Created directory: {target_directory}")
    except Exception as e:
        print(f"Error creating directory: {e}")
else:
    print(f"Directory already exists: {target_directory}")

# Step 3: Create 'Project' directory within /home/ubuntu
project_directory = os.path.join(target_directory, 'Project')
if not os.path.exists(project_directory):
    try:
        os.makedirs(project_directory)
        print(f"Created 'Project' directory: {project_directory}")
        os.chdir(project_directory)  # Move to the Project folder
    except Exception as e:
        print(f"Error creating 'Project' directory: {e}")
else:
    os.chdir(project_directory) # Move to the Project folder
    print(f"'Project' directory already exists: {project_directory}")

# Now your current working directory should be /home/ubuntu/Project
print(f"Current working directory: {os.getcwd()}")

#%%
# Import
import os
import pandas as pd
import sys
sys.path.insert(0, os.getcwd())
from Utils_Team2 import *  # Call functions as Utils
#os.listdir(os.getcwd())

#%%
################################################
# Load file (Option 1. Use Github(URL), 2. Use Cloud directory,
################################################
# 1. Job (2. Use Cloud directory)
path = '/home/ubuntu/Project/Data_cleaned'
df_job = pd.read_pickle(os.path.join(path, 'df_Occupation.pkl'))
print(df_job.shape)
# 2. Resume_previous one (2. Use Cloud directory)
df_resume = pd.read_pickle(os.path.join(path, 'resume_data_cleaned.pkl'))
print(df_resume.shape)
#%%
init_chk_df_2(df_resume)
#%%
# 2. Resume Update (1. Use Github(URL))
url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Amjad/Code/resumes_data.csv'
df_resume = pd.read_csv(url)
init_chk_df_2(df_resume)  #['ID', 'Resume', 'Category']
#%%
df_resume = df_resume[['Category', 'Resume']]
#%%
###############################################################################
# 1. Semantic Search using Siamese-BERT Networks (Sentence-BERT)
# BERT uses cross-encoder networks that take 2 sentences as input to the transformer network.
# Fine-tuned a pre-trained BERT network using Siamese and triplet network structures.
# adds a pooling operation to the output of BERT to derive a fixed-sized sentence embedding vector.
# The produced embedding vector is more appropriate for sentence similarity comparisons
#  within a vector space (i.e. that can be compared using cosine similarity).
###############################################################################

#%%
#!pip install sentence-transformers
# Load the BERT model.
# Various models trained on Natural Language Inference (NLI)
# https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/nli-models.md and
# Semantic Textual Similarity are available
# https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/sts-models.md
# code adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/application_semantic_search.py

from sentence_transformers import SentenceTransformer
import scipy.spatial

#%%
# 1. Setup an Embedding of Job description Corpus

#sentences = ['Absence of sanity','Lack of saneness','A man is eating food.']
sentences = df_job['Description_Job'].tolist()  #['Title','Description_Job']
# Each sentence is encoded as a 1-D vector with 78 columns
model = SentenceTransformer('bert-base-nli-mean-tokens')  # Test

sentence_embeddings = model.encode(sentences)
print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))
print('Sample BERT embedding vector' , sentence_embeddings[0]) #- note includes negative values'
#%%

model = SentenceTransformer('bert-base-nli-mean-tokens')  # Test
import numpy as np
# np.save('Job_sentence_embeddings.npy', sentence_embeddings) # Save to a .npy file
sentence_embeddings = np.load('Job_sentence_embeddings.npy')

#%%
# 2. Test: Perform Semantic Search on Resume description
import random
def get_random_resumes(category, num_samples):
    # Choose samples in a category
    df_category = df_resume[df_resume['Category'] == category]

    # If the category is not empty, select random resumes
    if not df_category.empty:
        random_indices = random.sample(list(df_category.index), min(num_samples, len(df_category)))
        queries = df_category.loc[random_indices, 'Resume']
        return queries
    else:
        return "No resumes found for this category."
#%%
#df_resume['Category'].value_counts()
df_resume['Category'].value_counts()
#%%
# Get Resume samples
query = get_random_resumes('SALES', 3)
#%%
query
#%%
# Parameter : Number of Jobs to recommend
number_top_matches = 5
#query = df_resume['Resume'].iloc[0]  # sample result
print("Semantic Search Results")
# Find the closest N sentences of the corpus for each query sentence based on cosine similarity
for idx,q in enumerate(query):
    queries = [q]
    query_embeddings = model.encode(queries)

    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n======================\n\n")
        print(f'Query: {idx}') #query
        print("\nTop 5 most recommendable Occupations:")

        for idx, distance in results[0:number_top_matches]:
            print(df_job.loc[idx, 'Title'], "(Cosine Score: %.4f)" % (1 - distance))  # Title of Job
    #        print(df_job[idx].strip(), "(Cosine Score: %.4f)" % (1-distance))  # Job Description
