
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tokenize import word_tokenize, MWETokenizer
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
# %%
# Read the CSV file
df_occupation = pd.read_csv("df_Occupation.csv")

# Display the first few rows of the DataFrame
print("Original DataFrame:")
print(df_occupation.head())
# %%
# Combining text columns into a single column
text_columns = ['Description', 'Description_Abilities',
                'Description_Knowledge', 'Description_Skills']
df_occupation['combined_text'] = df_occupation[text_columns].fillna(
    '').agg(' '.join, axis=1)

# %%
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# %%
# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_text = [lemmatizer.lemmatize(
        token) for token in tokens if token not in stopwords.words('english')]
    return ' '.join(cleaned_text)


# Apply preprocessing to the combined text column
df_occupation = df_occupation['combined_text'].apply(preprocess_text)

# %%

########### This seems better in cleaning importance than above #################
# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Example custom stopwords
custom_stopwords = set(['specific', 'additional', 'required'])
tokenizer = MWETokenizer()  # For phrase detection


def advanced_clean(text):

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [
        token for token in tokens if token not in stop_words and token not in custom_stopwords]
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures().chi_sq, 10)
    tokenizer.add_mwe(bigrams)
    tokens = tokenizer.tokenize(tokens)
    # tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tagged_tokens = nltk.pos_tag(tokens)
    tokens = [word for word, tag in tagged_tokens if tag in ('NN', 'VB')]

    return ' '.join(tokens)


# Apply advanced cleaning to each text entry
df_occupation['processed_text'] = df_occupation['combined_text'].apply(
    advanced_clean)

# %%
print(df_occupation['processed_text'])
# Get the minimum and maximum length of the texts
min_length = df_occupation['processed_text'].min()
max_length = df_occupation['processed_text'].max()
# compare
print(df_occupation['processed_text'][1])
print(df_occupation['combined_text'][1])
print(f"Minimum length: {min_length}")
print(f"Maximum length: {max_length}")

# %%
# why there are empty texts after cleaning?
# Calculate the length of each text entry then display min
# becuase it was not informing!!

min_length_row = df_occupation.loc[df_occupation['processed_text'].apply(
    len).idxmin()]
max_length_row = df_occupation.loc[df_occupation['processed_text'].apply(
    len).idxmax()]

# Display these rows
print("Row with Minimum Length:")
print(min_length_row)
print("\nRow with Maximum Length:")
print(max_length_row)
# %%
# maxumum number of characters

# Tokenize each entry in 'processed_text' and find the maximum length in words
max_length_words = df_occupation['processed_text'].apply(
    lambda x: len(word_tokenize(x))).max()

print(f"Maximum length in words for padding: {max_length_words}")

# %%
# Check for 'NA' values
na_counts = (df_occupation == 'NA').sum()
print(f"Dataset shape: {df_occupation.shape}")
print(f"Number of 'NA' in each column: {na_counts}")

# Replace NaN values with 'NA' in specific columns
df_occupation['Title'].fillna('NA', inplace=True)
df_occupation['processed_text'].fillna('NA', inplace=True)

# Delete all rows that have 'NA' in 'Title' or 'processed_text'
df_occupation = df_occupation[(df_occupation['Title'] != 'NA') & (
    df_occupation['processed_text'] != 'NA')]

# Check for NaN values again
nan_counts = df_occupation.isna().sum()
print(f"Number of NaN in each column after cleaning: {nan_counts}")
