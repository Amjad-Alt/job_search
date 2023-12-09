
#recommed a job title from job discription based on similarty with resume 
# Load your job descriptions dataset
url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation.csv'
df = pd.read_csv(url)
# Combining text columns into a single column
text_columns = ['Description', 'Description_Abilities',
                'Description_Knowledge', 'Description_Skills', 'Description_Tech','Description_Job']
df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)

# Handle missing values
df['Title'].fillna('NA', inplace=True)
df['combined_text'].fillna('NA', inplace=True)

# Creating a new DataFrame with only the relevant columns
job_df = df[['Title', 'combined_text']]


# Load pre-trained model tokenizer and model
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

import pickle

# Preprocess and save job encodings
job_encodings = {}
for index, row in job_df.iterrows():
    job_title = row['Title']
    job_description = row['combined_text']
    job_encoding = encode_text(job_description)
    job_encodings[job_title] = job_encoding

# Save the encodings to a file
with open('job_encodings.pkl', 'wb') as f:
    pickle.dump(job_encodings, f)
