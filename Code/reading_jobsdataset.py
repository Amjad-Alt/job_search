
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# Read the CSV file
df_occupation = pd.read_csv("df_Occupation.csv")

# Display the first few rows of the DataFrame
print("Original DataFrame:")
print(df_occupation.head())
#%%
# Combining text columns into a single column
text_columns = ['Description', 'Description_Abilities', 'Description_Knowledge', 'Description_Skills']
df_occupation['combined_text'] = df_occupation[text_columns].fillna('').agg(' '.join, axis=1)
#%%

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
#%%
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
#%%
# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_text = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    return ' '.join(cleaned_text)

# Apply preprocessing to the combined text column
df_occupation= df_occupation['combined_text'].apply(preprocess_text)

# %%
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
import re
#%%

########### This seems better in cleaning importance than above #################
# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_stopwords = set(['specific', 'additional', 'required'])  # Example custom stopwords
tokenizer = MWETokenizer()  # For phrase detection

def advanced_clean(text):
    
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and token not in custom_stopwords]
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures().chi_sq, 10)
    tokenizer.add_mwe(bigrams)
    tokens = tokenizer.tokenize(tokens)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tagged_tokens = nltk.pos_tag(tokens)
    tokens = [word for word, tag in tagged_tokens if tag in ('NN', 'VB')]
    
    return ' '.join(tokens)

# Apply advanced cleaning to each text entry
df_occupation['processed_text'] = df_occupation['combined_text'].apply(advanced_clean)

# %%
