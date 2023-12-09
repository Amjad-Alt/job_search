#%%
###############################################################################
# Environment: Cloud setting (will be cleaned or substituted when submit
###############################################################################

#%%
### Project folder on the Cloud
import os
# Step 1: Check current path
current_path = os.getcwd()
print(f"Current working directory: {current_path}")
# Step 2: Create /home/ubuntu if it doesn't exist
target_directory = '/home/ubuntu'
if not os.path.exists(target_directory):
    try:
        os.makedirs(target_directory)
        print(f"Created directory: {target_directory}")
    except Exception as e:
        print(f"Error creating directory: {e}")
else:
    print(f"Directory already exists: {target_directory}")

# Step 3: Create 'Project' directory within /home/ubuntu
project_directory = os.path.join(target_directory, 'Project')
if not os.path.exists(project_directory):
    try:
        os.makedirs(project_directory)
        print(f"Created 'Project' directory: {project_directory}")
        os.chdir(project_directory)  # Move to the Project folder
    except Exception as e:
        print(f"Error creating 'Project' directory: {e}")
else:
    os.chdir(project_directory) # Move to the Project folder
    print(f"'Project' directory already exists: {project_directory}")

# Now your current working directory should be /home/ubuntu/Project
print(f"Current working directory: {os.getcwd()}")

#%%
# Import
import os
import pandas as pd
import sys
sys.path.insert(0, os.getcwd())
from Utils_Team2 import *  # Call functions as Utils
#os.listdir(os.getcwd())

#%%
###############################################################################
# Data Source 1. Resume Database (Kaggle)
# https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
###############################################################################
#%%
# Resume Update (From Github repo)
url = r'https://raw.githubusercontent.com/Amjad-Alt/job_search/Amjad/Code/resumes_data.csv'
df_resume_2 = pd.read_csv(url)

#%%
init_chk_df_2(df_resume_2)
# Resume Update (From Github repo)

#%%
# # 1 (Kaggle resume)
# First download kaggle in mobaxtream termnal: pip install kaggle
# Then you will see kaggle "/home/ubuntu/.kaggle/"
# Then go to this: https://www.kaggle.com/settings
# Then press on "Create New Token" which is your auth keys in json file
# it will be downloaded in to your local
# move it Mobaextreme '.kaggle' and past it there
# so kaggle will work when you import it

#import kaggle # Let's try to change this path like our github url
#%%
# def download_dataset(dataset_name, destination_folder):
#     try:
#         kaggle.api.dataset_download_files(dataset_name, path=destination_folder, unzip=True)
#         print(f"Dataset downloaded to '{destination_folder}'")
#         return True
#     except Exception as e:
#         print(f"An error occurred while downloading the dataset: {e}")
#         return False
#
# def main():
#     dataset_name = 'gauravduttakiit/resume-dataset'
#     destination_folder = '/home/ubuntu/Project/resume'  # Complete the path
#
#     # Downloading the dataset
#     if download_dataset(dataset_name, destination_folder):
#         file_path = os.path.join(destination_folder, 'UpdatedResumeDataSet.csv')
#         df_resume = read_dataset(file_path)
#         if df_resume is not None:
#             df_resume_clean = clean_resume_data(df_resume)
#             initial_analysis(df_resume_clean)   DDoDo
#if __name__ == "__main__":
#   main()
#%%
#%%
### Resume folder on the Cloud
# Creating folder to store resume data inside Project:
# Define the path to the "Project" directory
project_directory = '/home/ubuntu/Project'
# Define the name of the folder to create
folder_name = 'resume'
# Create the "resume" folder inside the "Project" directory
resume_directory = os.path.join(project_directory, folder_name)
# Check if the folder already exists or create it
if not os.path.exists(resume_directory):
    os.makedirs(resume_directory)
    print(f"Created '{folder_name}' folder inside '{project_directory}'")
    os.chdir(resume_directory) # Move to the Project folder
else:
    print(f"'{folder_name}' folder already exists inside '{project_directory}'")
    os.chdir(resume_directory) # Move to the Project folder
print(f"Current working directory: {os.getcwd()}")
#%%
# Initial Analysis of the Resume Dataset
# First 5 rows:
#       Category                                             Resume
# 0  Data Science  Skills * Programming Languages: Python (pandas...
# 1  Data Science  Education Details  May 2013 to May 2017 B.E   ...
# 2  Data Science  Areas of Interest Deep Learning, Control Syste...
# 3  Data Science  Skills - R - Python - SAP HANA - Tableau - SAP...
# 4  Data Science  Education Details   MCA   YMCAUST,  Faridabad,...

#%%
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
#%%
# Saving the file to the Cloud on the cloned git repo
if __name__ == "__main__":
    main()

#%%
###############################################################################
#
# Data Source 2. O*NET® 28.0 Database (Can download on the WEB)
#
###############################################################################
#%%
###############################################################################
# Section 1. Load data (40 text files)
###############################################################################
#%%
import pandas as pd
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

#%%
def main():
    target_directory = '/home/ubuntu'
    project_directory = os.path.join(target_directory, 'Project')
    os.chdir(project_directory)  # Go back to the Project main
    # A. Define Download Link: Choose txt format (sql: create sql database)
    url = r'https://www.onetcenter.org/dl_files/database/db_28_0_text.zip'
    # B. Read Bulk data (*.zip file) and unzip to individual files
    open_url_zip(url)
    # Set the desired working directory path (ONET raw data)
    new_directory = "/home/ubuntu/Project/db_28_0_text/"
    # Change the current working directory
    os.chdir(new_directory)
    # Verify the new working directory
    print(f"Changed working directory to: {os.getcwd()}")
    # Search downloaded ONET files (*.txt)
    current_directory = os.getcwd()  # Get the current working directory
    files = os.listdir(current_directory)  # List files in the current working directory
    # Filter for files ending with '.txt'
    onet_raw = [file for file in files if file.endswith('.txt')]
    print(onet_raw)  # Print the list of matching files

    # 40 individual files
    print(len(onet_raw) - 1)  # Include ReadMe file
    onet_raw.sort()
#%%
# Downloading ONET files to the Cloud on the cloned git repo
if __name__ == "__main__":
    main()

#%%
###############################################################################
# Section 2.  Basic exploration and Preprocessing
#     1. Join Occupation name and description from Occupation Data
#     2. Join Element description from Content Model Reference
#     3. Join Scale description (Minimum, Maximum) from Content Scales Reference
###############################################################################
#%%
def main():
    '''# Base data- #1. Occupation Data:
      - 1016 unique jobs with code, title, description'''
    new_directory = "/home/ubuntu/Project/db_28_0_text/"
    # Change the current working directory
    os.chdir(new_directory)
    init_chk_df('Occupation Data.txt')  # Basic check of the file
    df = read('Occupation Data.txt')
    print(len(df['O*NET-SOC Code'].unique()))

    # Draft: Define 10 resume skill-related files
    skill_keywords = ['Abilities.txt', 'Interests.txt', 'Knowledge.txt', 'Skills.txt'
        , 'Technology Skills.txt', 'Tools Used.txt', 'Work Activities.txt', 'Work Context.txt'
        , 'Work Styles.txt', 'Work Values.txt']
    # A. Skimming datasets
    for file in skill_keywords:
        init_chk_df(file)

    # B. Preprocessing 1: Aggregate tables (Each category data + Base description columns)
    # Work on 5 tables among 10
    df = read('Abilities.txt') # read
    df = join_key(df)  # join columns
    init_chk_df_3(df)  #check features
    globals()['df_Abilities'] = df

    df = read('Interests.txt')
    df = join_key(df)
    init_chk_df_3(df)  #check features
    globals()['df_Interests'] = df

    df = read('Knowledge.txt')
    df = join_key(df)
    init_chk_df_3(df)
    globals()['df_Knowledge'] = df

    df = read('Skills.txt')
    df = join_key(df)
    init_chk_df_3(df)
    globals()['df_Skills'] = df

    df = read('Technology Skills.txt')
    # Different data structure (Only joined Job description)
    df = pd.merge(df, read('Occupation Data.txt'), how='left', on='O*NET-SOC Code')
    globals()['df_Tech_Skills'] = df
    return
#%%
if __name__ == "__main__":
    main()

#%%
###############################################################################
# Section 3. Make Job Description Corpus (for 1000 Occupations in ONET)
# 1. Make Job_Description corpus of each category datasets
# Rule (pay attention to Key elements in an Occupation)
#     Step 0. Start from the first Category (ex. Ability)
#     step 1. Choose High Demand Elements in an Occupation (Level Demand ratio over 50%, Pick Top 3)
#     step 2. Create Description column by an Occupation: (ex. Description_Abilities)
#     'Category: + 'Descriptions of High Demand elements
#     (ex) Abilities: The ability to A ...  The ability to B ... The ability to C ...

#     Step 3. Iterate for of 5 Categories : Get 5 Description columns
#     Description_Abilities, ... , Description_Interests
#
# 2. Make Final Job dataset with Full_Job_Description corpus (Merge of 5 Job_Description corpus)
#  Join to the Job Occupation dataset
# 'Job_Description' + 'Description_Abilities' .. + 'Description_Interests'
# (Add on Dec 7)
# 1. add Job_zone (for classification model)
# 2. prepare ref.data sets for inference(streamlit): Job zone, Related Jobs, Work sets
#  TBD: Think of other version of Job_data (size of Corpus)
#    (Current) Top 3 elements by each category > (Expand) Top 5, 10..
###############################################################################

#%%
#  1. Make Final Job dataset with Full_Job_Description corpus
datasets = [df_Abilities, df_Knowledge, df_Skills, df_Tech_Skills, df_Interests]
names = ['df_Abilities', 'df_Knowledge', 'df_Skills', 'df_Tech_Skills', 'df_Interests']
#temp = job_corpus(datasets[0], 0)

# 2. Make Final Job dataset with Full_Job_Description corpus
for idx,i in enumerate(datasets):
    df_names = ['df_Abilities', 'df_Knowledge', 'df_Skills', 'df_Tech_Skills', 'df_Interests']
    df_name = df_names[idx]
    globals()[df_name] = job_corpus(i, idx)

#%%
# Read Occupation dataset
# 1016 Jobs, O*NET-SOC Code, Title, Description
raw_onet_dir = "/home/ubuntu/Project/db_28_0_text/"
path = os.path.join(raw_onet_dir,'Occupation Data.txt')
df = read(path)
# Ref.Datasets
df_Job_Zone = read(os.path.join(raw_onet_dir,'Job Zones.txt'))
df_Job_Zone_Ref = read(os.path.join(raw_onet_dir,'Job Zone Reference.txt'))
df_relat_Jobs = read(os.path.join(raw_onet_dir,'Related Occupations.txt'))
#%%
# Join 6 Corpus datasets To the Occupation dataset
datasets = [df_Abilities, df_Knowledge, df_Skills, df_Tech_Skills, df_Interests]
df_names = ['df_Abilities', 'df_Knowledge', 'df_Skills', 'df_Tech_Skills', 'df_Interests']

for idx,i in enumerate(datasets):
    df_name = df_names[idx]
    field_name = df_name.split('_')[1]  # Extract category field from dataframe name
    key_col = ['O*NET-SOC Code', 'Description_' + field_name]
    print(df.shape)
    df = pd.merge(df, i[key_col], how='left', on='O*NET-SOC Code')
    print(df.shape)

#%%
df = df.fillna('')   # concatenate of NaN value with strings results in a NaN (So need to convert into blank)
# Make Full_Job_Description corpus Column
df['Description_Job'] = df['Title'] + ' ' + df['Description'] + ' ' + df['Description_Abilities'] \
                        + ' ' + df['Description_Knowledge'] + ' ' + df['Description_Skills'] \
                        + ' ' + df['Description_Tech'] + ' ' + df['Description_Interests']
#%%
# Add job zone column to Job data
print(df.shape)
df = pd.merge(df, df_Job_Zone[['O*NET-SOC Code','Job Zone']], how='left', on='O*NET-SOC Code')
print(df.shape)
#%%
#df.columns.to_list()

#%%
# df_Occupation_v2(12/9): Top 5 elements(function job_corpus), Add Job Title ahead
# save_as_pickle(df, '/home/ubuntu/Project/Data_cleaned', 'df_Occupation_v2.pkl') # Check Point
# df.to_csv("/home/ubuntu/Project/Data_cleaned/df_Occupation_v2.csv")
#%%
# # Final: Sample Check
# title = 'Statisticians'  #Dentists, General
# filter = (df['Title'] == title)
# print(df[filter][ ['Title', 'Description_Job']].to_string())
# print(len(df[filter]['Description_Job'].values[0])) #Length: 3343


#%%
#save_as_pickle(df, '/home/ubuntu/Project/Data_cleaned', 'df_Occupation.pkl') # Check Point
#%%
# path = '/home/ubuntu/Project/Data_cleaned'
# df_job = pd.read_pickle(os.path.join(path, 'df_Occupation.pkl')) #df_Occupation_v2
# print(df_job.shape)


#%%
# df_resume = pd.read_pickle(os.path.join(path, 'resume_data_cleaned.pkl'))
# print(len(df_resume.iloc[:1,1:2].values[0][0])) # 4744
#%%
#%%
# (Add on Dec 7)
# 1. add Job_zone (for classification model)
# 2. prepare ref.data sets for inference(streamlit): Job zone, Related Jobs, Work sets

# #%%
# # Ref.Datasets
# raw_onet_dir = "/home/ubuntu/Project/db_28_0_text/"
# df_Job_Zone = read(os.path.join(raw_onet_dir,'Job Zones.txt'))
# df_Job_Zone_Ref = read(os.path.join(raw_onet_dir,'Job Zone Reference.txt'))
# df_relat_Jobs = read(os.path.join(raw_onet_dir,'Related Occupations.txt'))
#%%
# save_as_pickle(df_Job_Zone, '/home/ubuntu/Project/Data_cleaned/JOB_ONET_data_pkl', 'df_Job_Zone.pkl')
# save_as_pickle(df_Job_Zone_Ref, '/home/ubuntu/Project/Data_cleaned/JOB_ONET_data_pkl', 'df_Job_Zone_Ref.pkl')
# save_as_pickle(df_relat_Jobs, '/home/ubuntu/Project/Data_cleaned/JOB_ONET_data_pkl', 'df_relat_Jobs.pkl')

#%%
# Add job zone
# print(df_job.shape)
# df_job = pd.merge(df_job, df_Job_Zone[['O*NET-SOC Code','Job Zone']], how='left', on='O*NET-SOC Code')
# print(df_job.shape)
# #%%
# os.getcwd()
#%%
#save_as_pickle(df_job, '/home/ubuntu/Project/Data_cleaned', 'df_Occupation.pkl') # Check Point
#df_job.to_csv("./df_Occupation.csv")

#%%
# Sample check (for identifying a certain job, info)
# filter = (df_Abilities['Title'] == 'Statisticians') & (df_Abilities['Element Name'] == 'Problem Sensitivity')
# sample = df_Abilities[filter]
# col = ['Title', 'Element Name', 'Scale Name', 'Minimum', 'Maximum', 'Data Value', 'Value_ratio']
# print(sample[col].to_string())

# filter = (df_Tech_Skills['Title'] == 'Statisticians') & (df_Tech_Skills['In Demand'] == 'Y')
# sample = df_Tech_Skills[filter]

#%%
# Check Point
#: Load Temporary Saved pickle DataFrames (saved 10 draft ONet datasets)

# Temporary : Save
# df.to_pickle(("./df_Work_Values.pkl"))
# df_Abilities = pd.read_pickle(("./df_Work_Values.pkl"))

# new_directory = "/home/ubuntu/Project/Data_cleaned/JOB_ONET_data_pkl"
# os.chdir(new_directory)
# print(f"Changed working directory to: {os.getcwd()}")
# #
# def search_df(format):
#     df_names =[]
#     for filename in os.listdir(os.getcwd()):
#         # check if the file is a .pkl file
#         if filename.endswith(format):
#             filepath = os.path.join(os.getcwd(), filename)
#             df_name = filename.split('.')[0]
#             df_names.append(df_name)
#             globals()[df_name] = pd.read_pickle(filepath)
#             print(f'Shape of {df_name}: {globals()[df_name].shape}')
#     return df_names
#
# df_names = search_df('.pkl')  # Load stored the dataframes
#%%
#%%
# Sample check
# df = df_Abilities
# col2 = ['Title', 'Element Name','Description_Ele','Data Value','Value_ratio']
# filter = (df['Title'] == 'Statisticians') & (df['Scale ID'] == 'LV')
# filter2 = (df['Title'] == title) & (df['Scale ID'] == 'LV') & (df['Value_ratio'] > 0.5)
# sample = df[filter][col2]
#
# # Distribution of elements' level values
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set_theme(style="darkgrid")
# sns.displot(sample, x="Value_ratio", kde = True)
# plt.tight_layout()
# plt.show()