from Utils_Team2 import *
import os
import sys
import pandas as pd
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, load_metric
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoConfig, DataCollatorWithPadding, AdamW, get_scheduler)
from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())

# %%



# Load and preprocess data

def create_zone_model_data():
    url = 'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation_v2.csv'
    df_job = pd.read_csv(url)
    df_job.dropna(subset=['Job Zone'], inplace=True)

    # Concatenate the text columns #
    text_columns = ['Description', 'Description_Abilities',
                    'Description_Knowledge', 'Description_Skills']

    # text_columns = ['Description', 'Description_Abilities' 'Description_Knowledge',
    #                 'Description_Skills', 'Description_Tech', 'Description_Interests']

    df_job['combined_text'] = df_job[text_columns].fillna(
        '').agg(' '.join, axis=1)

    df_job['Job Zone'] = df_job['Job Zone'].astype('int64') - 1
    df_job.rename(columns={'Job Zone': 'label'}, inplace=True)
    df_job = df_job[['combined_text', 'label']]
    return df_job


df_job = create_zone_model_data()
df_job.to_csv("./job_zone_model_data.csv")

# Load dataset
data = load_dataset("csv", data_files="./job_zone_model_data.csv")


# Preprocess dataset
def preprocess_data(data):
    '''Preprocessing of Transformer data'''
    # Convert the train dataset to a Pandas DataFrame
    df = data['train'].to_pandas()

    # Apply DataFrame operations
    df = df.drop_duplicates(subset=['combined_text'])
    df = df.reset_index(drop=True)[['combined_text', 'label']]

    # Convert back to Hugging Face dataset
    return Dataset.from_pandas(df)


data = preprocess_data(data)
data = data.train_test_split(test_size=0.3, seed=15)

# Tokenization
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len = 512


def tokenize(batch):
    return tokenizer(batch["combined_text"], truncation=True, max_length=512)


tokenized_dataset = data.map(tokenize, batched=True)
tokenized_dataset.set_format(
    "torch", columns=["input_ids", "attention_mask", "label"])


# Define Classifier
class Classifier(nn.Module):
    def __init__(self, pretrained_model, num_labels):
        super(Classifier, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(
            pretrained_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.pretrained_model(
            input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)


# Model setup
num_labels = 5
model = Classifier(AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(
    checkpoint)), num_labels).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Dataloader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=16, collate_fn=data_collator)
eval_dataloader = DataLoader(
    tokenized_dataset["test"], batch_size=16, collate_fn=data_collator)

# Training setup
num_epochs = 3
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                             num_training_steps=num_epochs * len(train_dataloader))

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to train and evaluate model
def train_and_evaluate(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, device, num_epochs):
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # Evaluation loop
    total_correct = 0
    total_predictions = 0
    model.eval()
    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs
        predictions = torch.argmax(logits, dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

    # Calculate accuracy
    return total_correct / total_predictions


# Hyperparameters for grid search
learning_rates = [1e-5, 3e-5, 5e-5]
batch_sizes = [8, 16]
num_epochs_options = [3, 4]



# Grid search
best_accuracy = 0
best_hyperparameters = {}

for lr, batch_size, num_epochs in itertools.product(learning_rates, batch_sizes, num_epochs_options):
    # DataLoaders for current batch size
    train_dataloader = DataLoader(
        tokenized_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
    eval_dataloader = DataLoader(
        tokenized_dataset["test"], batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))

    # Initialize model for current hyperparameters
    bert_model = AutoModel.from_pretrained(
        checkpoint, config=AutoConfig.from_pretrained(checkpoint))
    model = Classifier(bert_model, num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_epochs * len(train_dataloader))

    # Train and evaluate the model
    accuracy = train_and_evaluate(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, device, num_epochs)

    # Check if current model is the best one
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_hyperparameters = {'learning_rate': lr,
                                'batch_size': batch_size, 'num_epochs': num_epochs}

# Print the best hyperparameters and their corresponding accuracy
print(
    f"Best Hyperparameters: {best_hyperparameters}, Best Accuracy: {best_accuracy:.4f}")

# Define the path to save the model
# model_save_path = "./trained_model2.pth"

# Save the model's state dictionary
# torch.save(model.state_dict(), model_save_path)

best_model_path = "trained_model2.pth"  # Update this with the actual path

# Load the best model
best_model = Classifier(AutoModel.from_pretrained(
    checkpoint, config=AutoConfig.from_pretrained(checkpoint)), num_labels)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)
best_model.eval()

# Function to make predictions on the test set


def make_predictions(model, dataloader):
    model.eval()
    predictions = []
    real_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs
            preds = torch.argmax(logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            real_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(real_labels)


# Make predictions
predictions, real_labels = make_predictions(best_model, eval_dataloader)

# Calculate accuracy
accuracy = accuracy_score(real_labels, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Plot the confusion matrix
cm = confusion_matrix(real_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
