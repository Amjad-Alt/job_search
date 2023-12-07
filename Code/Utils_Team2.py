import pandas as pd
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import os
import pandas as pd
#import kaggle

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
        df = df.groupby(job_col).head(3) #leave the top 3 elements by each Job code
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
