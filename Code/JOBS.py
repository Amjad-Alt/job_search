import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import BertModel, AdamW

import torch
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
#%%

def get_bert_embedding(text, tokenizer, model):
    # Tokenize the input text and obtain the attention mask
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
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


# Assuming 'embeddings' and 'job_titles_encoded' are already defined
X = torch.stack(list(embeddings)).detach().numpy()
y = job_titles_encoded
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(
    X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(
    X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))


class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return self.classifier(pooled_output)


bert_model = BertModel.from_pretrained('bert-base-uncased')
model = BertClassifier(bert_model, num_labels=len(encoder.categories_[0]))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = batch
        input_ids, attention_mask = inputs[:,
                                           0], inputs[:, 1]  # Adjust if needed
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} finished, Average Loss: {avg_loss}")
#%%
