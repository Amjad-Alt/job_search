from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from transformers import BertModel
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Download necessary NLTK data (to be executed if needed)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')


def advanced_clean(text):
    """
    Cleans and tokenizes text data.

    Parameters:
    text (str): The text to be cleaned and tokenized.

    Returns:
    str: The cleaned and tokenized text.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [
        token for token in tokens if token not in stop_words and token not in custom_stopwords]
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures().chi_sq, 10)
    tokenizer.add_mwe(bigrams)
    tokens = tokenizer.tokenize(tokens)
    tagged_tokens = nltk.pos_tag(tokens)
    tokens = [word for word, tag in tagged_tokens if tag in ('NN', 'VB')]
    return ' '.join(tokens)


# Setup for text processing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_stopwords = set(['specific', 'additional', 'required'])
tokenizer = MWETokenizer()  # For phrase detection

# Load the CSV file
df_occupation = pd.read_csv("df_Occupation.csv")

# Combining text columns into a single column
text_columns = ['Description', 'Description_Abilities',
                'Description_Knowledge', 'Description_Skills']
df_occupation['combined_text'] = df_occupation[text_columns].fillna(
    '').agg(' '.join, axis=1)

# Apply advanced cleaning to each text entry
df_occupation['processed_text'] = df_occupation['combined_text'].apply(
    advanced_clean)

# Handle missing values
df_occupation['Title'].fillna('NA', inplace=True)
df_occupation['processed_text'].fillna('NA', inplace=True)
df_occupation = df_occupation[(df_occupation['Title'] != 'NA') & (
    df_occupation['processed_text'] != 'NA')]

# One-Hot Encoding for job titles (optional, depending on further use)
encoder = OneHotEncoder(sparse=False)
job_titles_encoded = encoder.fit_transform(
    np.array(df_occupation['Title']).reshape(-1, 1))

# Split the data for model training (optional, depending on further use)
X = df_occupation['processed_text']
y = job_titles_encoded
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Add model training and evaluation steps here (if applicable)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data


def tokenize_function(examples):
    return tokenizer(examples['processed_text'], padding='max_length', truncation=True)


# Tokenize all texts
tokenized_texts = df_occupation['processed_text'].apply(
    lambda x: tokenize_function({'processed_text': x}))

# Convert labels to numeric format
label_encoder = OneHotEncoder()  # or another suitable encoder
encoded_labels = label_encoder.fit_transform(
    df_occupation[['Title']]).toarray()

# Create a dataset class


class JobTitleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


# Instantiate the dataset
dataset = JobTitleDataset(tokenized_texts, encoded_labels)

# DataLoader creation here...

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load pre-trained BERT model for sequence classification
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.categories_[0]))


class BertForMultiLabelSequenceClassification(nn.Module):
    def __init__(self, num_labels):
        super(BertForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits


# Initialize model
model = BertForMultiLabelSequenceClassification(
    num_labels=len(label_encoder.categories_[0]))

# Training setup: loss function, optimizer, etc.
optimizer = AdamW(model.parameters(), lr=5e-5)


# Assuming train_loader and test_loader are already defined as DataLoaders

# Specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
epochs = 3

# Optimizer and learning rate scheduler
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * epochs  # Total number of training steps
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        # Ensure batch data is on the correct device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass adjustment
        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels.float())
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss}")

from sklearn.metrics import accuracy_score

# ...

# Validation loop
model.eval()
total_eval_accuracy = 0

for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)

    # Apply sigmoid to get probabilities and threshold to get predictions
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.5).int()  # Threshold, can be adjusted

    # Flatten the tensors to calculate accuracy across all labels
    predictions_flat = predictions.view(-1)
    labels_flat = labels.view(-1)

    total_eval_accuracy += accuracy_score(labels_flat.cpu().numpy(), predictions_flat.cpu().numpy())

avg_val_accuracy = total_eval_accuracy / len(test_loader)
print(f"Epoch {epoch+1} - Validation Accuracy: {avg_val_accuracy}")

# Save the model for deployment
# model.save_pretrained('./model')
