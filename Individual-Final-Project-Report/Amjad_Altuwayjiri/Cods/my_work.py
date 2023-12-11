# Packages needed
import pickle
import io
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoConfig
from Utils_Team2 import *
import pandas as pd
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoConfig,
                          DataCollatorWithPadding, AdamW, get_scheduler)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, plot_confusion_matrix

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
num_epochs_options = [3]

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

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_job['combined_text'], df_job['label'], test_size=0.3, random_state=42)

# Feature extraction with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# MLP Model
mlp = MLPClassifier()

# Define grid search parameters
param_grid = {
    'hidden_layer_sizes': [(50,), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

# Grid search
grid_search = GridSearchCV(mlp, param_grid, cv=3,
                           scoring='accuracy', verbose=2)
grid_search.fit(X_train, train_labels)

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate on test set
best_mlp_model = grid_search.best_estimator_

# Predict the test set results
y_pred = best_mlp_model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(test_labels, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Define range of sample sizes to plot learning curve
train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# Calculate learning curve scores for both training and test
train_sizes, train_scores, val_scores = learning_curve(
    best_mlp_model, X_train, train_labels, train_sizes=train_sizes, cv=5
)

# Calculate mean and standard deviation of training scores and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std,
                 val_mean + val_std, alpha=0.1)
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()


# Generate the confusion matrix plot
plot_confusion_matrix(best_mlp_model, X_test, test_labels, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# Define your custom Classifier class


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


# Define a function to load the trained model
def load_model(model_path, pretrained_model, num_labels):
    model = Classifier(pretrained_model, num_labels)
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model


# Logistic Regression Model
log_reg = LogisticRegression()

# Define grid search parameters
param_grid = {'C': [0.1, 1, 10], 'max_iter': [100, 200]}

# Grid search
grid_search = GridSearchCV(log_reg, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, train_labels)

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate on test set
best_model = grid_search.best_estimator_


# Define range of sample sizes to plot learning curve
train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# Calculate learning curve scores for both training and test
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, train_labels, train_sizes=train_sizes, cv=5
)

# Calculate mean and standard deviation of training scores and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std,
                 val_mean + val_std, alpha=0.1)
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()


# Generate the confusion matrix plot
plot_confusion_matrix(best_model, X_test, test_labels, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# Load the tokenizer for job level prediction
checkpoint = "bert-base-uncased"
tokenizer_for_prediction = AutoTokenizer.from_pretrained(checkpoint)

# Load your trained model for job level prediction
num_labels = 5  # Replace with the number of job level classes
bert_model_for_prediction = AutoModel.from_pretrained(
    checkpoint, config=AutoConfig.from_pretrained(checkpoint))
loaded_model = load_model("./trained_model.pth",
                          bert_model_for_prediction, num_labels)

# Streamlit app setup
st.title('We\'re looking for a job')
st.write('Hello, job seekers!')
st.markdown('<div class="title-box"><h1>First, upload your resume as a PDF file</h1></div>',
            unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type="pdf")

# Define the function to read and preprocess the PDF for job level prediction


def read_and_preprocess_for_job_level(file):
    with pdfplumber.open(file) as pdf:
        text = "\n".join([page.extract_text()
                         for page in pdf.pages if page.extract_text()])
    return text

# Define the function to preprocess text and make predictions for job level


def predict_job_level(text, model, tokenizer):
    # Tokenize the input text
    encoded_input = tokenizer(
        text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    # Run the model to get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Get the predicted class (job level)
    prediction = torch.argmax(outputs, dim=-1).item()
    return prediction


def get_job_level_info(prediction):
    job_level_info = {
        0: ("Intern", "Little or no previous work-related skill, knowledge, or experience is needed for these occupations. "
                      "For example, a person can become a waiter or waitress even if he/she has never worked before."),
        1: ("Entry Level", "Some previous work-related skill, knowledge, or experience is usually needed. For example, a teller "
                           "would benefit from experience working directly with the public."),
        2: ("Junior", "Previous work-related skill, knowledge, or experience is required for these occupations. For example, "
                      "an electrician must have completed three or four years of apprenticeship or several years of vocational "
                      "training, and often must have passed a licensing exam, in order to perform the job."),
        3: ("Mid Level", "A considerable amount of work-related skill, knowledge, or experience is needed for these occupations. "
                         "For example, an accountant must complete four years of college and work for several years in accounting "
                         "to be considered qualified."),
        4: ("Senior", "Extensive skill, knowledge, and experience are needed for these occupations. Many require more than five "
                      "years of experience. For example, surgeons must complete four years of college and an additional five to "
                      "seven years of specialized medical training to be able to do their job."),
    }

    # Return the corresponding job level information (title and description)
    return job_level_info.get(prediction, ("Unknown Level", "No description available"))


if uploaded_file is not None:
    resume_text_for_job_level = read_and_preprocess_for_job_level(
        io.BytesIO(uploaded_file.getvalue()))
    st.write('Contents of the uploaded PDF file:')
    st.text_area("Resume Text", resume_text_for_job_level, height=300)

    if st.button('Predict Job Level'):
        job_level = predict_job_level(
            resume_text_for_job_level, loaded_model, tokenizer_for_prediction)
        job_level_title, job_level_description = get_job_level_info(job_level)
        st.write(f'**Recommended Job Level to Apply For:** {job_level_title}')
        st.write(job_level_description)


# Load the precomputed job encodings
with open('job_encodings.pkl', 'rb') as f:
    job_encodings = pickle.load(f)

# Recommend jobs based on uploaded file
if uploaded_file is not None and st.button('Recommend Jobs'):
    resume_text_for_recommendation = read_and_preprocess_for_job_level(
        io.BytesIO(uploaded_file.getvalue()))
    # resume_encoding = encode_text(
    #     resume_text_for_recommendation, tokenizer_for_recommendation, model_for_recommendation)

    resume_encoding = encode_resume_text(resume_text_for_recommendation)

    # Compute similarities
    scores = {}
    for job_title, job_encoding in job_encodings.items():
        score = np.dot(resume_encoding, job_encoding) / (np.linalg.norm(
            resume_encoding) * np.linalg.norm(job_encoding))  # Cosine similarity
        scores[job_title] = score

    # Display job recommendations
    recommended_jobs = sorted(scores, key=scores.get, reverse=True)[:5]
    st.write("**Top job recommendations based on your resume:**")
    for job in recommended_jobs:
        st.write(job)
