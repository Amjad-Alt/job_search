#%%
import os
os.chdir(os.path.join('/home/ubuntu', 'Project'))
#%%
# Import
import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.getcwd())
from Utils_Team2 import *  # Call functions as Utils
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
#%%
# Create data from a scratch
url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation_v2.csv'
temp = create_zone_model_data(url)
temp['label'] = temp['label'].astype(int) - 1
temp.to_csv("./job_zone_model_data.csv") # to_csv, for load_dataset()
data=load_dataset("csv",data_files="./job_zone_model_data.csv") #Transformer format: DatasetDict
data = prep_zone_model_train_data(data) #923, train:
data = train_test_split(data,0.3) # 70% train, 30 % Test
#%%
data
#%%
train_texts = data['train']['Description_Job']
val_texts = data['test']['Description_Job']
train_labels = torch.tensor([int(label) for label in data['train']['label']])
val_labels = torch.tensor([int(label) for label in data['test']['label']])
#%%
train_labels
#%%
# GPU(device), Define Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

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
#test_dataset = Dataset(test_encodings, test_labels)

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
#%%
from transformers import get_linear_schedule_with_warmup

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
# Define the scheduler
num_training_steps = len(train_dataset) * training_args.num_train_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_training_steps)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer) #fly to the maximum length within each batch
#data_collator = DataCollatorForLanguageModeling( tokenizer=tokenizer, mlm=True, mlm_probability=0.15) #masking model

from datasets import load_metric
metric = load_metric("accuracy") # Define the metric

# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Create the Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer, scheduler),
    data_collator=data_collator,  # Pass the data collator here
    compute_metrics=compute_metrics
)
trainer.train()
#%%
# Evaluate the model
eval_results = trainer.evaluate()
# Print the evaluation results
for key, value in eval_results.items():
    print(f"{key}: {value}")
#%%
model_name = "light_tune_RoBERTa"
model_path = os.path.join(os.getcwd(), model_name)
trainer.save_model(model_path)