from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import AdamW,  get_scheduler,  AutoTokenizer
import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import itertools

# Load the CSV file
df_occupation = pd.read_csv("df_Occupation.csv")

text_columns = ['Description', 'Description_Abilities', 'Description_Knowledge', 'Description_Skills']
df_occupation['combined_text'] = df_occupation[text_columns].fillna('').agg(' '.join, axis=1)

# Handle missing values
df_occupation['Title'].fillna('NA', inplace=True)
df_occupation['combined_text'].fillna('NA', inplace=True)

# Replace NaN values in 'Job Zone_x' with a placeholder (e.g., -1 or a similar non-conflicting value)
df_occupation['Job Zone_x'].fillna(-1, inplace=True)

# Filter out rows where 'Job Zone_x' is NaN (or the placeholder if NaNs are replaced)
df_occupation = df_occupation[df_occupation['Job Zone_x'] != -1]

# Split the data for model training
X = df_occupation['combined_text']
y = df_occupation['Job Zone_x']

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    X, y, test_size=0.2, random_state=42)
#%%

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(texts):
    return tokenizer(texts, truncation=True, padding='max_length', max_length=570)

# Apply tokenization to the texts
train_encodings = tokenize_function(train_texts.tolist())
test_encodings = tokenize_function(test_texts.tolist())


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


#%%

# Model
model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name)
pretrained_model = AutoModel.from_pretrained(model_name, config=config)


class Classifier(nn.Module):
    def __init__(self, pretrained_model, num_labels):
        super(Classifier, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(
            input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


num_labels = len(np.unique(y))
model = Classifier(pretrained_model, num_labels)

#%%
# Hyperparameters for grid search
learning_rates = [1e-5, 3e-5, 5e-5]
batch_sizes = [16, 32]
num_epochs = 3
# Define a function for model training and evaluation
def train_and_evaluate(model, train_dataloader, test_dataloader, learning_rate, criterion, lr_scheduler):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Training loop
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = criterion(outputs, batch['labels'].long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    # Evaluation loop
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == batch['labels']).sum().item()
            total_samples += batch['labels'].size(0)

    # Calculate and return total accuracy
    total_accuracy = total_correct / total_samples
    return total_accuracy

# %%

# Grid search
best_accuracy = 0
best_hyperparameters = {}

for lr, batch_size in itertools.product(learning_rates, batch_sizes):
    # Initialize dataloaders with the current batch size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = Classifier(pretrained_model, num_labels)

    # Set up criterion and scheduler
    criterion = nn.CrossEntropyLoss()
    num_training_steps = 3 * len(train_dataloader) # Assuming 3 epochs
    lr_scheduler = get_scheduler("linear", optimizer=AdamW(model.parameters(), lr=lr), 
                                 num_warmup_steps=0, num_training_steps=num_training_steps)

    # Train and evaluate the model
    accuracy = train_and_evaluate(model, train_dataloader, test_dataloader, lr, criterion, lr_scheduler)

    # Update best model if current model is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_hyperparameters = {'learning_rate': lr, 'batch_size': batch_size}

print("Best Hyperparameters:", best_hyperparameters)
# %%
