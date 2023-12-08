

import torch
from transformers import BertTokenizer, BertModel
#%%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
#%%

def get_bert_embedding(text, tokenizer, model):
    # Tokenize the input text and obtain the attention mask
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        # from previous cleaning script and addded 10 just for me :)
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Forward pass, get the hidden states from BERT's last layer
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    # Obtain the embeddings from the last hidden layer
    embeddings = last_hidden_states[0][:, 0, :].squeeze()
    return embeddings
# %%

embeddings = df_occupation['processed_text'].apply(
    lambda x: get_bert_embedding(x, tokenizer, model))

# Convert embeddings to tensor
#embeddings = torch.stack(list(embeddings)).numpy()

# %%
