

from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# %%

def get_bert_embeddings(text_list):
    # Ensure text_list is a list of strings
    if not isinstance(text_list, list):
        text_list = [text_list]
    inputs = tokenizer(text_list, return_tensors='pt',
                       padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings
# %%

batch_size = 5  # Adjust based on your computational capacity
embeddings = []

for i in range(0, len(cleaned_resumes), batch_size):
    batch = cleaned_resumes[i:i+batch_size].tolist()
    batch_embeddings = get_bert_embeddings(batch)
    embeddings.append(batch_embeddings)

# Flatten the list of embeddings
embeddings = torch.cat(embeddings, dim=0)
# %%

# Apply the function to each resume
# Process resumes
resume_embeddings = cleaned_resumes.apply(get_bert_embeddings)

# Process job descriptions
job_description_embeddings = cleaned_job_descriptions.apply(
    get_bert_embeddings)
