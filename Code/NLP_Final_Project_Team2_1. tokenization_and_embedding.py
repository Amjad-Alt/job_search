#%%
# Importing the dataset from the cloud:

import pandas as pd

# Define the paths to the pickle files
abilities_path = "/home/ubuntu/job_search/Data_cleaned/df_Abilities.pkl"
interests_path = "/home/ubuntu/job_search/Data_cleaned/df_Interests.pkl"
knowledge_path = "/home/ubuntu/job_search/Data_cleaned/df_Knowledge.pkl"
skills_path = "/home/ubuntu/job_search/Data_cleaned/df_Skills.pkl"
tools_used_path = "/home/ubuntu/job_search/Data_cleaned/df_Tools_Used.pkl"
work_activities_path = "/home/ubuntu/job_search/Data_cleaned/df_Work_Activities.pkl"
work_context_path = "/home/ubuntu/job_search/Data_cleaned/df_Work_Context.pkl"
work_styles_path = "/home/ubuntu/job_search/Data_cleaned/df_Work_Styles.pkl"
work_values_path = "/home/ubuntu/job_search/Data_cleaned/df_Work_Values.pkl"
resume_data_path = "/home/ubuntu/job_search/Data_cleaned/resume_data_cleaned.pkl"

# Function to load a pickle file
def load_pickle(file_path):
    try:
        return pd.read_pickle(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Load each dataset
df_abilities = load_pickle(abilities_path)
df_interests = load_pickle(interests_path)
df_knowledge = load_pickle(knowledge_path)
df_skills = load_pickle(skills_path)
df_tools_used = load_pickle(tools_used_path)
df_work_activities = load_pickle(work_activities_path)
df_work_context = load_pickle(work_context_path)
df_work_styles = load_pickle(work_styles_path)
df_work_values = load_pickle(work_values_path)
df_resume_data = load_pickle(resume_data_path)

# Testing the dataframes:
print(df_abilities.head())
print(df_resume_data)

#
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import torch

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and truncate/pad resumes
max_length = 512  # BERT's maximum sequence length
tokenized = df_resume_data['Resume'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=max_length, truncation=True))

# Padding and creating attention masks
padded = pad_sequences(tokenized, maxlen=max_length, padding='post', truncating='post')
attention_mask = np.where(padded != 0, 1, 0)

# Convert to PyTorch tensors
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

# Get embeddings
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()

# Compute similarity
cosine_similarity_matrix = cosine_similarity(embeddings)

