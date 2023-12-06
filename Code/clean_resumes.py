

import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
df = pd.read_csv("resumes_data.csv")

# display the first few rows of the DataFrame
print(df.head())
# %%
# Select the 'Resume' column
resumes = df['Resume']
print(resumes.head())

# %%
# clean
# %%
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# %%
# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove line breaks and extra white spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    cleaned_text = [lemmatizer.lemmatize(
        token) for token in tokens if token not in stopwords.words('english')]
    return ' '.join(cleaned_text)


# Apply cleaning to each resume
cleaned_resumes = resumes.apply(clean_text)

# Print first few cleaned resumes
print(cleaned_resumes.head())
# %%
