import torch
from transformers import BertTokenizer, BertModel

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text_list):
    # Ensure text_list is a list of strings
    if not isinstance(text_list, list):
        text_list = [text_list]

    inputs = tokenizer(text_list, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

# Batch processing for BERT embeddings
batch_size = 5  # Adjust based on your computational capacity
embeddings = []

for i in range(0, len(resumes_with_id), batch_size):
    batch = resumes_with_id["Cleaned_Resume"][i:i+batch_size].tolist()
    batch_embeddings = get_bert_embeddings(batch)
    embeddings.append(batch_embeddings)

# Flatten the list of embeddings
embeddings = torch.cat(embeddings, dim=0)

# %%
# Apply the function to each resume
# Process resumes
resume_embeddings = resumes_with_id["Cleaned_Resume"].apply(get_bert_embeddings)
# Process job descriptions
job_description_embedding = get_bert_embeddings(job_description)

# %%
