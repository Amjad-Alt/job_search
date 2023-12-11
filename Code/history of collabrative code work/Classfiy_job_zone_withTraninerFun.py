#%%
import sys
import os
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict, load_metric
from transformers import (
    BertForSequenceClassification, Trainer, TrainingArguments,
    BertTokenizer, BertConfig, DataCollatorWithPadding
)

# Add the directory containing Utils_Team2.py to sys.path
sys.path.append('/home/ubuntu/job_search/Code/')


# Change the current working directory
os.chdir('/home/ubuntu/job_search/Code/')

# Import custom utility functions
from Utils_Team2 import *

#%%
# Load datasets from URLs
url_job = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation.csv'
df_job = pd.read_csv(url_job)

# Preprocessing function for job zone model data
def create_zone_model_data(df):
    df.dropna(subset=['Job Zone'], inplace=True)
    df['Job Zone'] = df['Job Zone'].astype('int64') - 1
    df.rename(columns={'Job Zone': 'label'}, inplace=True)
    return df[['Description_Job', 'label']]

df_job_processed = create_zone_model_data(df_job)
df_job_processed.to_csv("./job_zone_model_data.csv")

# Load and preprocess the dataset
data_files = "./job_zone_model_data.csv"
raw_datasets = load_dataset("csv", data_files=data_files)

#%%

# Define the model name and load the tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

#%%
def tokenize(batch):
    return tokenizer(batch['Description_Job'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = raw_datasets.map(tokenize, batched=True)

split_datasets = tokenized_datasets['train'].train_test_split(test_size=0.3)
dataset_dict = DatasetDict({
    'train': split_datasets['train'],
    'test': split_datasets['test']
})

config = BertConfig.from_pretrained(model_name, num_labels=5)

model = BertForSequenceClassification.from_pretrained(
    model_name,
    config=config
)

accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")

#%%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"],
        "recall": recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"],
    }

#%%
# Tranining loop
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("./model_output")


