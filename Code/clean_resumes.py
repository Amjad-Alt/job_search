
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Read the CSV file
df = pd.read_csv("resumes_data.csv")

# Display the first few rows of the DataFrame
print("Original DataFrame:")
print(df.head())

# Extract 'ID' and 'Resume' columns
resumes_with_id = df[['ID', 'Resume']]

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # Lowercase, remove line breaks, extra white spaces, and punctuation
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize, remove stopwords, and lemmatize
    tokens = word_tokenize(text)
    cleaned_text = [lemmatizer.lemmatize(
        token) for token in tokens if token not in stopwords.words('english')]
    return ' '.join(cleaned_text)


# Apply cleaning to each resume
resumes_with_id['Cleaned_Resume'] = resumes_with_id['Resume'].apply(clean_text)

# Print first few cleaned resumes with IDs
print("Cleaned Resumes with IDs:")
print(resumes_with_id[['ID', 'Cleaned_Resume']].head())

