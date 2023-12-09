import pandas as pd
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import os
#import kaggle
from datasets import load_dataset,Dataset,DatasetDict
from transformers import DataCollatorForLanguageModeling
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from sentence_transformers import SentenceTransformer
import scipy.spatial
import torch
import torch.nn as nn
import pandas as pd
#%%
# ################################################
# [Classification] Model to Predict Job zone (with Job Corpus)
# Model 1. Fine-tuning  roberta-based Model
#
# ################################################

def create_zone_model_data():
    ''''
    # Create data to make Classification Model (Ready to Load by Transformer)
    '''
    # Load current data (cleaned ONET JOB Corpus)
    url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation.csv'
    df_job = pd.read_csv(url)
    # Explore Target # ['O*NET-SOC Code',  'Description_Job', 'Job Zone']
    # . 1~5, Nan (9%)
    # : wide range of characteristics which do not fit into one of the detailed O*NET-SOC occupations.
    print("Original")
    print(df_job['Job Zone'].value_counts(normalize=True, dropna=False).sort_index())
    # df_job[df_job['Job Zone'].isnull()==1].Title
    df_job.dropna(subset=['Job Zone'], inplace=True)
    # df_job['Job Zone'].isnull().sum()
    df_job['Job Zone'] = df_job.loc[:, 'Job Zone'].astype('int64')  #change label type to INT
    print("After Cleansing NaN Label")
    df_job['Job Zone'] = df_job['Job Zone'] -1  #Align to Transformer (label should start from 0)
    print(df_job['Job Zone'].value_counts(normalize=True, dropna=False).sort_index())
    df_job.rename(columns={'Job Zone': 'label'}, inplace=True)
    df_job = df_job[['Description_Job', 'label']]
    return df_job

def prep_zone_model_train_data(data):
    '''Preprocessing of Transformer data'''
    data = data.remove_columns(['Unnamed: 0'])
    data.set_format('pandas')
    data = data['train'][:]
    data.drop_duplicates(subset=['Description_Job'], inplace=True)
    data = data.reset_index()[['Description_Job', 'label']]
    data = Dataset.from_pandas(data)
    return data

def train_test_split(data, test_size):
    # train(70%), test(30%)
    train_test = data.train_test_split(test_size=test_size, seed=15)
    # gather as a single DatasetDict
    data = DatasetDict({
        'train': train_test['train'],
        'test': train_test['test']})
    return data

def tokenize(batch):
    # Define pretrained model
    checkpoint = "cardiffnlp/twitter-roberta-base-emotion"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_len = 512
    return tokenizer(batch["Description_Job"], truncation=True,max_length=512)

class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(CustomModel, self).__init__()
        self.num_labels = num_labels

        # Load Model with given checkpoint and extract its body
        self.model = model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint,
                                                                                                     output_attentions=True,
                                                                                                        output_hidden_states=True))
        #The Dropout layer: used to prevent overfitting
        self.dropout = nn.Dropout(0.1)  # 10%: set to zero when training
        self.classifier = nn.Linear(768, num_labels)  # load and initialize weights
        # 768: Bert Base (# of hidden layers, w 12 attention layers)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        # CLS token
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        #- Loss Function
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Option: BCELoss(): Binary classification
            #loss_fct = nn.BCELoss()

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

def create_zone_model_test_resume(): # 2481 resumes
    url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Amjad/Code/resumes_data.csv'
    df_resume = pd.read_csv(url)  #2484
    # Clean Null description
    df_resume.dropna(subset=['Resume'], inplace=True)  # 2483
    df_resume = df_resume.drop_duplicates(subset=['Resume']) # 2481
    return df_resume # Keep df_resume (for analysis/application later)

def prep_zone_model_test_resume(data):
    '''Create test data same as training data (Preprocessing, tokenize)'''
    data = data.remove_columns(['Unnamed: 0'])
    data.set_format('pandas')
    data = data['train'][:]
    #data = data['train'][:50]  # Sample for test
    data = Dataset.from_pandas(data)
    return data

def tokenize_resume(batch):
    checkpoint = "cardiffnlp/twitter-roberta-base-emotion"  # Pretrained model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_len = 512
    return tokenizer(batch["Resume"], truncation=True,max_length=512)

def predict(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)
        # Flatten prediction tensor before appending to the list (ex. 40 with 3 batches: 16,16,8)
        prediction = prediction.view(-1)
        predictions.append(prediction)
        # predictions.extend(pred.tolist())  # Convert tensor to list and extend
        # predictions = np.array(predictions)  # Convert list to numpy array
    predictions = torch.cat(predictions) #Concatenate the list of tensors into one tensor
    predictions = predictions.tolist()
    return predictions
#%%
##################################################
# Similarity base Job recommendation
##################################################
# 1. Semantic Search using Siamese-BERT Networks (Sentence-BERT)
def Create_Embedding_corpus(df, corpus):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentences = df[corpus].tolist()
    sentence_embeddings = model.encode(sentences)
    print('Length BERT embedding vector:', len(sentence_embeddings[0]))
    print('Sample BERT embedding vector:', sentence_embeddings[0])  # includes negative values
    return sentence_embeddings


# 2. Test: Perform Semantic Search on Resume description
import random
def get_random_resumes(df,category, num_samples):
    # Choose samples in a category
    df_category = df[df['Category'] == category]
    # If the category is not empty, select random resumes
    if not df_category.empty:
        random_indices = random.sample(list(df_category.index), min(num_samples, len(df_category)))
        queries = df_category.loc[random_indices, 'Resume']
        return queries
    else:
        return "No resumes found for this category."

def Get_Job_Recommendation_Semantic_Similarity(df, category, num_samples, number_recommends):
    query = get_random_resumes(df, category, num_samples)  # Create N samples
    print("Semantic Search Results")
    # Find the closest N sentences of the corpus for each query sentence based on cosine similarity
    model = SentenceTransformer('bert-base-nli-mean-tokens')  # Test
    for idx, q in enumerate(query):
        queries = [q]
        query_embeddings = model.encode(queries)
        # Calculate Semantic_Similarity between resume and job corpus
        for query, query_embedding in zip(queries, query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            print("\n\n======================\n\n")
            print(f'Query: {idx}')  # sample # of query, can change it to whole resume: query
            print("\nTop 5 most recommendable Occupations:")
            url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation.csv'
            df_job = pd.read_csv(url)

            for idx, distance in results[0:number_recommends]:
                print(df_job.loc[idx, 'Title'], "(Cosine Score: %.4f)" % (1 - distance))  # Title of Job
#%%
##################################################
# Load and Preprocessing ONET data
##################################################

def open_url_zip(url):
    zip = urlopen(url)  #request.urlopen(url)
    # Extract zip file
    with ZipFile(BytesIO(zip.read())) as zfile:
        zfile.extractall()
    return

# Function: Read Individual file and show basic information
def read(file):
    path = os.path.join(os.getcwd(), file)
    df = pd.read_csv(path, sep='\t')
    return df

def create_zone_model_data():
    # Load Preprocessed Job data and preprocess it for modeling
    url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation.csv'
    df_job = pd.read_csv(url)
    # Explore Target # ['O*NET-SOC Code',  'Description_Job', 'Job Zone']
    # . 1~5, Nan (9%)
    # : wide range of characteristics which do not fit into one of the detailed O*NET-SOC occupations.
    print("Original")
    print(df_job['Job Zone'].value_counts(normalize=True, dropna=False).sort_index())
    # df_job[df_job['Job Zone'].isnull()==1].Title
    df_job.dropna(subset=['Job Zone'], inplace=True)
    # df_job['Job Zone'].isnull().sum()
    print("After Cleansing NaN Label")
    print(df_job['Job Zone'].value_counts(normalize=True, dropna=False).sort_index())
    df_job.rename(columns={'Job Zone': 'label'}, inplace=True)
    df_job = df_job[['Description_Job', 'label']]
    return df_job

def save_as_pickle(df, path, filename):
    df.to_pickle((os.path.join(path, filename)))

def search_df(format):
    '''
    Load format (ex. .pkl) file in the current directory
    Make dataframe with same file name and
    '''
    df_names =[]
    for filename in os.listdir(os.getcwd()):
        # check if the file is a .pkl file
        if filename.endswith(format):
            filepath = os.path.join(os.getcwd(), filename)
            df_name = filename.split('.')[0]
            df_names.append(df_name)
            globals()[df_name] = pd.read_pickle(filepath)
            print(f'Shape of {df_name}: {globals()[df_name].shape}')
    return df_names

def init_chk_df(file):
    df = read(file)
    print(f'Below is the basic information of the dataframe {file}\n')
    print(f'A.Data has {df.shape[0]} records with {df.shape[1]} features\n')
    print(f'B.First 5 sample records:\n{df.head(5)}\n')
    print(f'C.Features:\n{df.dtypes}\n')
    print(f'D.Null value check:\n{df.isnull().sum()}\n') #.isna() : same function (boolean)
    print(f'E.Summary statistics:\n{df.describe()}')
    cols = df.columns.to_list()
    return cols

def init_chk_df_2(df):
    print(f'A.Data has {df.shape[0]} records with {df.shape[1]} features\n')
    print(f'B.First 5 sample records:\n{df.head(5)}\n')
    print(f'C.Features:\n{df.dtypes}\n')
    print(f'D.Null value check:\n{df.isnull().sum()}\n') #.isna() : same function (boolean)
    print(f'E.Summary statistics:\n{df.describe()}')
    cols = df.columns.to_list()
    return cols

def init_chk_df_3(df):
    # Check diversity of ONET datasets
    print(len(df['O*NET-SOC Code'].unique()))  # 873 unique occupations
    print(len(df['Scale ID'].unique()), '\n')  # 2 models
    print(df['Scale Name'].value_counts(normalize=True).sort_index(), '\n')  # IM, LV
    print(len(df['Element Name'].unique()))  # 52 Abilities
    print(df['Element Name'].unique())

def value_ratio(df):
    df['Value_ratio'] = df['Data Value']/df['Maximum']
    return df

# Initial preprocessing of individual ONET data
def join_key(df):
    '''
    1. Join Occupation name and description from Occupation Data
    2. Join Element description from Content Model Reference
    3. Join Scale description (Minimum, Maximum) from Content Scales Reference
    :return: aggregated df
    '''
    print(df.shape)
    df = pd.merge(df, read('Occupation Data.txt'), how='left', on='O*NET-SOC Code')
    df = pd.merge(df, read('Content Model Reference.txt')[['Element ID', 'Description']], how='left', on= 'Element ID')
    df = pd.merge(df, read('Scales Reference.txt'), how='left', on='Scale ID')
    df.rename(columns={'Description_x': 'Description_SOC', 'Description_y': 'Description_Ele'}, inplace=True)
    value_ratio(df)
    print(df.shape)
    return df

import copy
def job_corpus(df, idx):
    df = copy.deepcopy(df)
    df_names = ['df_Abilities', 'df_Knowledge', 'df_Skills', 'df_Tech_Skills', 'df_Interests']
    df_name = df_names[idx]
    field_name = df_name.split('_')[1] # Extract category field from dataframe name
    scale_names = ['LV', 'LV', 'LV', '', 'OI'] # 'LV', df_Tech_Skills: none, Interest: 'OI'
    if df_name != 'df_Tech_Skills':
        # 1. Filter Level scale > 50%
        #filter = (df['Scale ID'] == scale_names[idx]) & (df['Value_ratio'] > 0.5)
        #df = df[filter]
        # 2. Choose Top 3 High Demand Elements in an Occupation
        job_col = ['O*NET-SOC Code', 'Title', 'Description_SOC']
        sort_rule = ['O*NET-SOC Code', 'Value_ratio', 'Element ID']
        ele_col = ['Element Name', 'Description_Ele']
        df = df.sort_values(sort_rule, ascending=[True, False, True])
        df = df.groupby(job_col).head(5) #leave the top 3 > 5elements by each Job code
        # 3. Merge 'Description_Ele' of 3 elements into 'Description_top_ele'
        df['Description_top_ele'] = df.groupby(job_col)['Description_Ele'].transform(lambda x: ' '.join(x))

    elif df_name == 'df_Tech_Skills':
        # 1. df_Tech_Skills Filter 'In Demand = Y': need for the particular occupation
        filter = (df['In Demand'] == 'Y')
        df = df[filter]
        # 2. Merge  'Description_Ele' of All Examples (Keyword) in an Occupation (No need to pick 3)
        df.rename(columns={'Description': 'Description_SOC'}, inplace=True) #Align with other data
        job_col = ['O*NET-SOC Code', 'Title', 'Description_SOC'] #No 'Description_SOC' in Tech_Skills
        df['Description_top_ele'] = df.groupby(job_col)['Example'].transform(lambda x: ' '.join(x))

    # 4. Finalize 'Description_{category}' corpus
    # category field: Job description A + Job description B + Job description C
    df['Description'] = field_name + ':' + df['Description_top_ele']
    df.rename(columns={'Description': 'Description_' + field_name}, inplace=True)
    #Description_Tech
    # 5. Clean category data (Dedup by Occupation key)
    key_col = ['O*NET-SOC Code','Description_' + field_name]
    print(df.shape)
    df = df.sort_values(by='Title')  # sort (first)
    df = df.drop_duplicates(subset=['Title'])  # dup remove
    df = df.reset_index(drop=True)  # reset index
    df = df[key_col]
    print(df.shape)
    return df

##################################################
# Load and Preprocessing RESUME data
##################################################

# Inactive: Kaggle api
def read_dataset(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"CSV file not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None

def clean_resume_data(df):
    # Replace newline characters and other formatting issues
    df['Resume'] = df['Resume'].str.replace(r'\r\n', ' ', regex=True)
    df['Resume'] = df['Resume'].str.replace(r'â¢', '-', regex=True)
    df['Resume'] = df['Resume'].str.replace(r'â¢', '*', regex=True)
    # Further cleaning steps can be added here
    return df

def initial_analysis(df):
    print(f"Dataset shape: {df.shape}")
    print(f"First 5 rows:\n{df.head()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Null values in each column:\n{df.isnull().sum()}")
    print(f"Unique categories:\n{df['Category'].value_counts()}")


def main():
    # Define the path to the downloaded CSV file
    csv_file_path = '/home/ubuntu/Project/resume/UpdatedResumeDataSet.csv'

    # Load the CSV file into a DataFrame
    print("Attempting to load dataset...")
    df = read_dataset(csv_file_path)

    if df is not None:
        print("Cleaning data...")
        df_cleaned = clean_resume_data(df)  # Assuming you have this function

        # Save the cleaned DataFrame as a pickle
        destination_folder = '/home/ubuntu/Project/Data_cleaned'
        file_name = 'resume_data_cleaned.pkl'
        print("Attempting to save DataFrame as pickle...")
        save_as_pickle(df_cleaned, destination_folder, file_name)
        print("Complete saving DataFrame as pickle")
    else:
        print("Failed to load dataset.")
