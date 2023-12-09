#%%
# Import
import os
import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import scipy.spatial
import sys
sys.path.insert(0, os.getcwd())
from Utils_Team2 import *  # Call functions as Utils
#%%
################################################
# Load file (Option 1. Use Github(URL), 2. Use Cloud directory,
################################################
#%%
# 1. Job (CSV 1. Use Github(URL))
url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation.csv'
df_job = pd.read_csv(url)

# 2. Resume Update (1. Use Github(URL))
url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Amjad/Code/resumes_data.csv'
df_resume = pd.read_csv(url)
#init_chk_df_2(df_resume)  #['ID', 'Resume', 'Category']
# 2.Use Cloud directory)
# path = '/home/ubuntu/Project/Data_cleaned'
# df_job = pd.read_pickle(os.path.join(path, 'df_Occupation.pkl'))
# df_resume = pd.read_pickle(os.path.join(path, 'resume_data_cleaned.pkl'))
#%%
df_resume.columns.to_list()
#%%
df_1 = df_job[['Title', 'Description_Job']]
df_2 = df_resume[['Category', 'Resume']]
# Create a temporary key for merging
df_1['key'] = 0
df_2['key'] = 0
# Perform an outer merge on the key, which gives the Cartesian product
combined_df = pd.merge(df_1, df_2, on='key', how='outer')
combined_df.drop('key', axis=1, inplace=True) # Drop the key as it's no longer needed
combined_df.columns = ['Title', 'Description_Job', 'Category', 'Resume']
#%%
combined_df.shape  #(2523744, 4)

#%%
# Define a threshold for similarity
threshold = 0.7
# Assign label based on similarity
label = 1 if similarity.item() > threshold else 0
print(f"Label: {label}")
# Cosine Similarity: 0.9502490758895874
# Label: 1