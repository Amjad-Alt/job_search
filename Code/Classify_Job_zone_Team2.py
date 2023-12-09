#%%
# Standard library imports
import os
import sys

# Modify sys.path to include necessary directories
sys.path.insert(0, os.getcwd())
sys.path.append('/home/ubuntu/job_search/Code/')

# Change the current working directory
os.chdir('/home/ubuntu/job_search/Code/')

# Import custom utility functions from Utils_Team2
from Utils_Team2 import *

# Third-party imports for data handling and machine learning
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Huggingface's transformers and datasets library
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoConfig, AutoTokenizer,
                          DataCollatorForLanguageModeling, DataCollatorWithPadding)
from transformers.modeling_outputs import TokenClassifierOutput

from datasets import load_dataset, Dataset, DatasetDict


#%%
################################################
# [Classification] Model to Predict Job zone (with Job Corpus)
# Model 1. Fine-tuning  roberta-based Model
# huggingface (twitter-roberta-base-emotion)
################################################
#%%
###############################################
# [Outline of Fine-tuning- Customized Network]
#    A. Inpt data Preprocessing (Use origin Tokenizer)
#    B. ★ Build CustomModel
#       - Extract BODY + Add New Head
#       : Add Dropout Layer, Dense layer for 5 categories
#    C. Train with Customized module
###############################################
#%%
################################################
# Load file (Option 1. Use Github(URL), 2. Use Cloud directory,
################################################
# 1. Job (CSV 1. Use Github(URL))
url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation.csv'
df_job = pd.read_csv(url)

# 2. Resume (New) (1. Use Github(URL))
url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Amjad/Code/resumes_data.csv'
df_resume = pd.read_csv(url)
#init_chk_df_2(df_resume)  #['ID', 'Resume', 'Category']

# 2.Use Cloud directory)
# path = '/home/ubuntu/Project/Data_cleaned'
# df_job = pd.read_pickle(os.path.join(path, 'df_Occupation.pkl'))
# df_resume = pd.read_pickle(os.path.join(path, 'df_New_resume_job_zone_pred.pkl')) # with job-zone predict
# #old resume data(moe's):resume_data_cleaned

#%%
# Definition of Job zone
# df_Job_Zone_Ref = read(os.path.join('/home/ubuntu/Project/db_28_0_text/','Job Zone Reference.txt'))
# # ['Job Zone', 'Name', 'Experience', 'Education', 'Job Training', 'Examples', 'SVP Range']
# print(df_Job_Zone_Ref[['Job Zone', 'Name']].to_string()) # Job zone 1~5 (easy ~ difficult)

#%%
################################################
# Preprocessing
################################################
#%%
def create_zone_model_data():
    ''''
    # Create data to make Classification Model (Ready to Load by Transformer)
    '''
    # Load current data (cleaned ONET JOB Corpus)
    url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation.csv'
    df_job = pd.read_csv(url)
    # Explore Target # ['O*NET-SOC Code',  'Description_Job', 'Job Zone']
    # . 1~5, Nan (9%)
    # : wide range of characteristics which do not fit into one of the detailed O*NET-SOC occupations.
    print("Original")
    print(df_job['Job Zone'].value_counts(normalize=True, dropna=False).sort_index())
    # df_job[df_job['Job Zone'].isnull()==1].Title
    df_job.dropna(subset=['Job Zone'], inplace=True)
    # df_job['Job Zone'].isnull().sum()
    df_job['Job Zone'] = df_job.loc[:, 'Job Zone'].astype('int64')  #change label type to INT
    print("After Cleansing NaN Label")
    df_job['Job Zone'] = df_job['Job Zone'] -1  #Align to Transformer (label should start from 0)
    print(df_job['Job Zone'].value_counts(normalize=True, dropna=False).sort_index())
    df_job.rename(columns={'Job Zone': 'label'}, inplace=True)
    df_job = df_job[['Description_Job', 'label']]
    return df_job
#%%
create_zone_model_data().to_csv("./job_zone_model_data.csv") # to_csv, for load_dataset()
#%%
# Transformer format: DatasetDict
data=load_dataset("csv",data_files="./job_zone_model_data.csv")

#%%
def prep_zone_model_train_data(data):
    '''Preprocessing of Transformer data'''
    data = data.remove_columns(['Unnamed: 0'])
    data.set_format('pandas')
    data = data['train'][:]
    data.drop_duplicates(subset=['Description_Job'], inplace=True)
    data = data.reset_index()[['Description_Job', 'label']]
    data = Dataset.from_pandas(data)
    return data

#%%
data = prep_zone_model_train_data(data) #923, train:
#%%
# Split train, evaluation dataset (70%, 30%)
def train_test_split(data, test_size):
    # train(70%), test(30%)
    train_test = data.train_test_split(test_size=test_size, seed=15)
    # gather as a single DatasetDict
    data = DatasetDict({
        'train': train_test['train'],
        'test': train_test['test']})
    return data
#%%
data = train_test_split(data,0.3)

#%%
################################################
# 1. Define Training Module
################################################
#%%
# Define pretrained model
checkpoint = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len=512
#%%
def tokenize(batch):
  return tokenizer(batch["Description_Job"], truncation=True,max_length=512)

tokenized_dataset = data.map(tokenize, batched=True)
tokenized_dataset
#%%
tokenized_dataset['train'][0]
#%%
tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label"])
#data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#Option: used in BERT(language modeling and applies random masking)
#data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
#%%
tokenized_dataset['train'][0] #need to check torch= tensor
#%%
class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(CustomModel, self).__init__()
        self.num_labels = num_labels

        # Load Model with given checkpoint and extract its body
        self.model = model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint,
                                                                                                     output_attentions=True,
                                                                                                        output_hidden_states=True))
        #The Dropout layer: used to prevent overfitting
        self.dropout = nn.Dropout(0.1)  # 10%: set to zero when training
        self.classifier = nn.Linear(768, num_labels)  # load and initialize weights
        # 768: Bert Base (# of hidden layers, w 12 attention layers)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        # CLS token
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        #- Loss Function
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Option: BCELoss(): Binary classification
            #loss_fct = nn.BCELoss()

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

#%%
# GPU(device), Define Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CustomModel(checkpoint=checkpoint,num_labels=5).to(device)
#%%
#  (Define) Dataloader
batch_size = 16
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator
)
#%%
# Set Hyperparameters,
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr=5e-5
## Optimizer
from transformers import AdamW,get_scheduler
optimizer = AdamW(model.parameters(), lr=lr)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

#%%
# Define Evaluate metric (classification)
from datasets import load_metric
metric = load_metric("accuracy") #multi class
#metric = load_metric("f1")

#%%
################################################
# 2. Training and Evaluate (execute modules)
################################################
#%%

# (Define) Training(w train_dataloader)/Evaluate(w test_dataloader)/Training loop (Epoch)
from tqdm.auto import tqdm

progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))

for epoch in range(num_epochs):
    model.train()  #Call train function
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch) # train
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar_train.update(1)

    model.eval() #Call eval function
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar_eval.update(1)

    print(metric.compute())  # check performance on validation set

#%%M
# Save Pytorch Model
# Model 1 : {'accuracy': 0.6137184115523465}
model_path = "./Model_Classify_Job_zone.pt"  # specify the path to save the model
torch.save(model.state_dict(), model_path)
model.load_state_dict(torch.load(model_path))

#%%
################################################
# 3. Job Zone classify model Test on Resume Data (for application)
################################################
#%%
def create_zone_model_test_resume(): # 2481 resumes
    url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Amjad/Code/resumes_data.csv'
    df_resume = pd.read_csv(url)  #2484
    # Clean Null description
    df_resume.dropna(subset=['Resume'], inplace=True)  # 2483
    df_resume = df_resume.drop_duplicates(subset=['Resume']) # 2481
    return df_resume # Keep df_resume (for analysis/application later)
#%%
df_resume = create_zone_model_test_resume()
df_resume['Resume'].to_csv("./job_zone_model_resume_test.csv")
# Transformer format: DatasetDict
data=load_dataset("csv",data_files="./job_zone_model_resume_test.csv")
#%%
def prep_zone_model_test_resume(data):
    '''Create test data same as training data (Preprocessing, tokenize)'''
    data = data.remove_columns(['Unnamed: 0'])
    data.set_format('pandas')
    data = data['train'][:]
    #data = data['train'][:50]  # Sample for test, # TBD: randomly choose sample in dataloader
    data = Dataset.from_pandas(data)
    return data
#%%
data = prep_zone_model_test_resume(data)  #2481
#%%
data
#%%
checkpoint = "cardiffnlp/twitter-roberta-base-emotion" # Pretrained model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len=512
#%%
def tokenize_resume(batch):
  return tokenizer(batch["Resume"], truncation=True,max_length=512)
#%%
tokenized_dataset = data.map(tokenize_resume, batched=True)
#%%
tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask"])
#%%
#tokenized_dataset[0]
#tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label"]) #Trainset
#%%
def predict(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)
        # Flatten prediction tensor before appending to the list (ex. 40 with 3 batches: 16,16,8)
        prediction = prediction.view(-1)
        predictions.append(prediction)
        # predictions.extend(pred.tolist())  # Convert tensor to list and extend
        # predictions = np.array(predictions)  # Convert list to numpy array
    predictions = torch.cat(predictions) #Concatenate the list of tensors into one tensor
    predictions = predictions.tolist()
    return predictions
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./Model_Classify_Job_zone.pt"  # Load Path of the Saved model 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CustomModel(checkpoint=checkpoint,num_labels=5).to(device) # CustomModel: Set trained model 1
model.load_state_dict(torch.load(model_path))
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Resume Dataloader, batch_size 16(train), NO shuffle (not train)
test_dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)
#%%
results = predict(model, test_dataloader, device)
# Job zone : Label + 1 (Label was modified to align with Transformer (start from 0): label = job zone -1)
results_ = [x + 1 for x in results]
#%%
print(len(results)) #2481
#%%
np.save('zone_model_test_resume.npy', results) # array save (Bult results on Resume data)
results = np.load('zone_model_test_resume.npy')
#%%
# Freq
from collections import Counter
print(Counter(results_))  # {4: 1707, 3: 555, 5: 214, 2: 5}

# Job Zone                                            Name
# 0         1   Job Zone One: Little or No Preparation Needed
# 1         2           Job Zone Two: Some Preparation Needed
# 2         3       Job Zone Three: Medium Preparation Needed
# 3         4  Job Zone Four: Considerable Preparation Needed
# 4         5     Job Zone Five: Extensive Preparation Needed
#%%
# Combine Predicted Job zone with an Original Resume data
df_resume['Job_Zone_pred'] = results_
df_resume.tail()
#%%
# # Distribution of elements' level values
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set_theme(style="darkgrid")
# sns.displot(df_resume, x="Job_Zone_pred", kde = True)
# plt.tight_layout()
# plt.show()
#%%
# save_as_pickle(df_resume, '/home/ubuntu/Project/Data_cleaned', 'df_New_resume_job_zone_pred.pkl') # Check Point
# df_resume.to_csv("./resumes_data_zone_pred.csv")

