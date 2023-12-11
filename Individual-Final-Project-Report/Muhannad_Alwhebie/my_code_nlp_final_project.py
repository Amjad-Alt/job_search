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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%
###############################################################################
# Data Source 1. Resume Database (Kaggle)
# https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
###############################################################################
#%%
# Resume Update (From Github repo)
url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Amjad/Code/resumes_data.csv'
df_resume_2 = pd.read_csv(url)

#%%
init_chk_df_2(df_resume_2)
# Resume Update (From Github repo)

#%%
# # 1 (Kaggle resume)
# First download kaggle in mobaxtream termnal: pip install kaggle
# Then you will see kaggle "/home/ubuntu/.kaggle/"
# Then go to this: https://www.kaggle.com/settings
# Then press on "Create New Token" which is your auth keys in json file
# it will be downloaded in to your local
# move it Mobaextreme '.kaggle' and past it there
# so kaggle will work when you import it

#import kaggle # Let's try to change this path like our github url
#%%
# def download_dataset(dataset_name, destination_folder):
#     try:
#         kaggle.api.dataset_download_files(dataset_name, path=destination_folder, unzip=True)
#         print(f"Dataset downloaded to '{destination_folder}'")
#         return True
#     except Exception as e:
#         print(f"An error occurred while downloading the dataset: {e}")
#         return False
#
# def main():
#     dataset_name = 'gauravduttakiit/resume-dataset'
#     destination_folder = '/home/ubuntu/Project/resume'  # Complete the path
#
#     # Downloading the dataset
#     if download_dataset(dataset_name, destination_folder):
#         file_path = os.path.join(destination_folder, 'UpdatedResumeDataSet.csv')
#         df_resume = read_dataset(file_path)
#         if df_resume is not None:
#             df_resume_clean = clean_resume_data(df_resume)
#             initial_analysis(df_resume_clean)   DDoDo
#if __name__ == "__main__":
#   main()
#%%
#%%
### Resume folder on the Cloud
# Creating folder to store resume data inside Project:
# Define the path to the "Project" directory
project_directory = '/home/ubuntu/Project'
# Define the name of the folder to create
folder_name = 'resume'
# Create the "resume" folder inside the "Project" directory
resume_directory = os.path.join(project_directory, folder_name)
# Check if the folder already exists or create it
if not os.path.exists(resume_directory):
    os.makedirs(resume_directory)
    print(f"Created '{folder_name}' folder inside '{project_directory}'")
    os.chdir(resume_directory) # Move to the Project folder
else:
    print(f"'{folder_name}' folder already exists inside '{project_directory}'")
    os.chdir(resume_directory) # Move to the Project folder
print(f"Current working directory: {os.getcwd()}")
#%%
# Initial Analysis of the Resume Dataset
# First 5 rows:
#       Category                                             Resume
# 0  Data Science  Skills * Programming Languages: Python (pandas...
# 1  Data Science  Education Details  May 2013 to May 2017 B.E   ...
# 2  Data Science  Areas of Interest Deep Learning, Control Syste...
# 3  Data Science  Skills - R - Python - SAP HANA - Tableau - SAP...
# 4  Data Science  Education Details   MCA   YMCAUST,  Faridabad,...

#%%
def main():
    # Define the path to the downloaded CSV file
    csv_file_path = '/home/ubuntu/Project/resume/UpdatedResumeDataSet.csv'

    # Load the CSV file into a DataFrame
    print("Attempting to load dataset...")
    df = read_dataset(csv_file_path)

    if df is not None:
        print("Cleaning data...")
        df_cleaned = clean_resume_data(df)  # Assuming you have this function

        # Save the cleaned DataFrame as a pickle
        destination_folder = '/home/ubuntu/Project/Data_cleaned'
        file_name = 'resume_data_cleaned.pkl'
        print("Attempting to save DataFrame as pickle...")
        save_as_pickle(df_cleaned, destination_folder, file_name)
        print("Complete saving DataFrame as pickle")
    else:
        print("Failed to load dataset.")
#%%
# Saving the file to the Cloud on the cloned git repo
if __name__ == "__main__":
    main()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 

 import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt


# Load and preprocess data
def create_zone_model_data():
    url = 'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation.csv'
    df_job = pd.read_csv(url)
    df_job.dropna(subset=['Job Zone'], inplace=True)

    # Concatenate the text columns
    text_columns = ['Description', 'Description_Abilities', 'Description_Knowledge', 'Description_Skills']
    df_job['combined_text'] = df_job[text_columns].fillna('').agg(' '.join, axis=1)

    df_job['Job Zone'] = df_job['Job Zone'].astype('int64') - 1
    df_job.rename(columns={'Job Zone': 'label'}, inplace=True)
    df_job = df_job[['combined_text', 'label']]
    return df_job

# Preprocess data
df_job = create_zone_model_data()

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df_job['combined_text'], df_job['label'], test_size=0.3, random_state=42)

# Feature extraction with TF-IDF

vectorizer = TfidfVectorizer(max_features=3000)  # Reduced max_features for potential better generalization

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Naive Bayes Model
nb_model = MultinomialNB()

# Define grid search parameters for Naive Bayes

param_grid = {'alpha': np.linspace(0.01, 1, 20)}  # Using finer steps in alpha to find the optimal regularization parameter

# Stratified K-Fold for handling class imbalances
stratified_k_fold = StratifiedKFold(n_splits=5)

# Grid search for Naive Bayes
grid_search = GridSearchCV(nb_model, param_grid, cv=stratified_k_fold, scoring='accuracy')

param_grid = {'alpha': [0.1, 1, 10]}

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate on test set
best_model = grid_search.best_estimator_

# Make predictions on test set
test_predictions = best_model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(test_labels, test_predictions)
print("Accuracy on Test Set:", accuracy)

# Define range of sample sizes to plot learning curve
train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# Calculate learning curve scores for both training and test
train_sizes, train_scores, val_scores = learning_curve(best_model, X_train, train_labels, train_sizes=train_sizes, cv=5)

# Calculate mean and standard deviation of training scores and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()

# Generate the confusion matrix plot
plot_confusion_matrix(best_model, X_test, test_labels, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()