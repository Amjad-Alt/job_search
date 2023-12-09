#%%
# Import
import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.getcwd())
from Utils_Team2 import *  # Call functions as Utils
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
#%%
# Create data from a scratch

df_job = pd.read_pickle(os.path.join('/home/ubuntu/Project/Data_cleaned', 'df_Occupation_v2.pkl'))


create_zone_model_data().to_csv("./job_zone_model_data.csv") # to_csv, for load_dataset()
data=load_dataset("csv",data_files="./job_zone_model_data.csv") #Transformer format: DatasetDict
data = prep_zone_model_train_data(data) #923, train:
data = train_test_split(data,0.3) # 70% train, 30 % Test
#%%
data['train']['Description_Job']
#%%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# def tokenize(batch):
#   return tokenizer(batch["Description_Job"], truncation=True,max_length=512)
#
# tokenized_dataset = data.map(tokenize, batched=True)
#tokenized_dataset
#%%
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

#%%
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

#%%
# Define a PyTorch Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)
test_dataset = Dataset(test_encodings, test_labels)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create the Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
trainer.save_model(os.getcwd())