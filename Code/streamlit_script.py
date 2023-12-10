import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import streamlit as st
import pdfplumber
import io
import pickle
import numpy as np

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


# Load the tokenizer for job level prediction
checkpoint = "bert-base-uncased"
tokenizer_for_prediction = AutoTokenizer.from_pretrained(checkpoint)

# Load your trained model for job level prediction
num_labels = 6  # Replace with the number of job level classes
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
    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    # Run the model to get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Get the predicted class (job level)
    prediction = torch.argmax(outputs, dim=-1).item()
    return prediction


# Process uploaded file for job level prediction when the button is clicked
if uploaded_file is not None:
    resume_text_for_job_level = read_and_preprocess_for_job_level(
        io.BytesIO(uploaded_file.getvalue()))
    st.write('Contents of the uploaded PDF file:')
    st.text_area("Resume Text", resume_text_for_job_level, height=300)

    if st.button('Predict Job Level'):
        job_level = predict_job_level(
            resume_text_for_job_level, loaded_model, tokenizer_for_prediction)
        st.write(f'Recommended Job Level to Apply For: {job_level}')

# Load tokenizer and model for job recommendations
tokenizer_for_recommendation = AutoTokenizer.from_pretrained(
    'bert-base-uncased')
model_for_recommendation = AutoModel.from_pretrained('bert-base-uncased')

# Define function to encode text for job recommendations


def encode_text(text, tokenizer, model):
    input_ids = tokenizer.encode(
        text, add_special_tokens=True, max_length=512, truncation=True)
    input_ids = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs[0][0].mean(dim=0).numpy()  # Mean pooling


# Load the precomputed job encodings
with open('job_encodings.pkl', 'rb') as f:
    job_encodings = pickle.load(f)

# Recommend jobs based on uploaded file
if uploaded_file is not None and st.button('Recommend Jobs'):
    resume_text_for_recommendation = read_and_preprocess_for_job_level(
        io.BytesIO(uploaded_file.getvalue()))
    resume_encoding = encode_text(
        resume_text_for_recommendation, tokenizer_for_recommendation, model_for_recommendation)

    # Compute similarities
    scores = {}
    for job_title, job_encoding in job_encodings.items():
        score = np.dot(resume_encoding, job_encoding) / (np.linalg.norm(
            resume_encoding) * np.linalg.norm(job_encoding))  # Cosine similarity
        scores[job_title] = score

    # Display job recommendations
    recommended_jobs = sorted(scores, key=scores.get, reverse=True)[:5]
    st.write("Top job recommendations based on your resume:")
    for job in recommended_jobs:
        st.write(job)
