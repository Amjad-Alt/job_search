import streamlit as stls
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pdfplumber
import io
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Custom CSS to inject
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

# Load your model (Update this with your actual model loading code)
# model = load('path/to/your/model.joblib')

nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return " ".join(filtered_tokens)


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
