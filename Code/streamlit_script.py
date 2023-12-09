import torch
from transformers import BertTokenizer, BertModel
import streamlit as stls
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pdfplumber
import io
from transformers import BertTokenizer
import sys
import streamlit as st
import pickle
# %%
st.markdown("""
    <style>
    .title-box {
        background-color: lightblue;
        padding: 7px;
        border-radius: 5px;
    }
    .title-box h1 {
        color: black;
        font-size:22px; /* Smaller font size */
    }
    </style>
""", unsafe_allow_html=True)


st.title('We\'re looking for a job')

st.write('Hello, job seekers!')

# Using a div with custom class for styling
st.markdown('<div class="title-box"><h1>First, upload your resume as a PDF file</h1></div>',
            unsafe_allow_html=True)

# File uploader widget
uploaded_file = st.file_uploader(" ", type="pdf")


def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return "\n".join(pages)


def clean_text(text):
    return text.strip()


if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    text = read_pdf(io.BytesIO(bytes_data))
    cleaned_text = clean_text(text)

    st.write('Contents of the uploaded PDF file:')
    st.text_area(" ", cleaned_text, height=300)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocess_text(text):
    encoded_input = tokenizer(
        text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    return encoded_input

# Load your model (Update this with your actual model loading code)
# model = load('path/to/your/model.joblib')


def predict_job_level(text):

    # Use your model to predict the job level
    # prediction = model.predict([processed_text])
    # return prediction

    # Placeholder return
    return "Senior Level"  # Replace this with actual prediction logic


if uploaded_file is not None:
    # ... [Code to read and clean the PDF text] ...

    preprocessed_text = preprocess_text(cleaned_text)

    # Predict the job level
    # Update this line with your prediction logic
    job_level = predict_job_level(preprocessed_text)

    # Display the prediction
    st.write('Recommended Job Level to Apply For:')
    st.write(job_level)

# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def encode_text(text):
    # Encode text to BERT's format
    input_ids = tokenizer.encode(
        text, add_special_tokens=True, max_length=512, truncation=True)
    input_ids = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs[0][0].mean(dim=0).numpy()  # Mean pooling


# Load the precomputed job encodings
with open('job_encodings.pkl', 'rb') as f:
    job_encodings = pickle.load(f)

if uploaded_file is not None:
    # Process the resume
    resume_text = read_pdf(io.BytesIO(uploaded_file.getvalue()))
    resume_encoding = encode_text(resume_text)

    # Compute similarities
    scores = {}
    for job_title, job_encoding in job_encodings.items():
        score = np.dot(resume_encoding, job_encoding) / (np.linalg.norm(
            resume_encoding) * np.linalg.norm(job_encoding))  # Cosine similarity
        scores[job_title] = score

    # Recommend job
    recommended_jobs = sorted(scores, key=scores.get, reverse=True)[
        :5]  # Top 5 recommendations
    st.write(f"Top job recommendations based on your resume:")
    for job in recommended_jobs:
        st.write(job)


# %%
