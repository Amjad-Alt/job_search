#%%
###############################################################################
# Source. O*NET® 28.0 Database
# Used API to get the raw data (40 text files), can also download on the WEB
# https://www.onetcenter.org/database.html
###############################################################################
# Import
import pandas as pd
import os
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import sys
sys.path.insert(0, os.getcwd())
from Utils_Team2 import *  # Call functions as Utils

def main():
    print('1. Read Bulk data (*.zip file) and unzip to individual files\n')
    url = r'https://www.onetcenter.org/dl_files/database/db_28_0_text.zip'
    open_url_zip(url)
    # Set the desired working directory path (ONET raw data)
    new_directory = "./db_28_0_text/"
    os.chdir(new_directory)  # Change the current working directory
    # Search downloaded ONET files (*.txt)
    current_directory = os.getcwd()  # Get the current working directory
    print('These are ONET DB files\n')
    files = os.listdir(current_directory)  # List files in the current working directory
    # Filter for files ending with '.txt'
    onet_raw = [file for file in files if file.endswith('.txt')] # 40 individual files
    onet_raw.sort()
    print(f'number of files: {len(onet_raw) - 1}')  # Include ReadMe file
    print('1. Get Occupation DB\n')
    init_chk_df('Occupation Data.txt')  # Basic check of the file

    print('2. Generate Corpus of each Key data Categories\n')
    print('A. Preprocessing 1: Aggregate tables (Each category data + Base description columns)\n')
    # Work on 5 tables among 10
    df = read('Abilities.txt') # read
    df = join_key(df)  # join columns
#    init_chk_df_3(df)  #check features
    globals()['df_Abilities'] = df
    print('Finished creation of Ability Corpus\n')

    df = read('Interests.txt')
    df = join_key(df)
#    init_chk_df_3(df)  #check features
    globals()['df_Interests'] = df
    print('Finished creation of Interests Corpus\n')

    df = read('Knowledge.txt')
    df = join_key(df)
#    init_chk_df_3(df)
    globals()['df_Knowledge'] = df
    print('Finished creation of Knowledge Corpus\n')

    df = read('Skills.txt')
    df = join_key(df)
#    init_chk_df_3(df)
    globals()['df_Skills'] = df
    print('Finished creation of Skills Corpus\n')

    df = read('Technology Skills.txt')
    # Different data structure (Only joined Job description)
    df = pd.merge(df, read('Occupation Data.txt'), how='left', on='O*NET-SOC Code')
    globals()['df_Tech_Skills'] = df
    print('Finished creation of technology Skills Corpus\n')

    print('B. Preprocessing 2: Create Job Corpus of each Category\n')
    datasets = [df_Abilities, df_Knowledge, df_Skills, df_Tech_Skills, df_Interests]
    df_names = ['df_Abilities', 'df_Knowledge', 'df_Skills', 'df_Tech_Skills', 'df_Interests']
    # 2. Make Final Job dataset with Full_Job_Description corpus
    for idx, i in enumerate(datasets):
        df_name = df_names[idx]
        globals()[df_name] = job_corpus(i, idx)  # Gather Top 5 important elements in each category
    print('-- End of Generating individual Corpus of each Categories --\n')
    return

if __name__ == "__main__":
    main()

print('3. [Final] Create Aggregated Job Corpus based on Occupation dataset\n')
df_job = read('Occupation Data.txt')
datasets = [df_Abilities, df_Knowledge, df_Skills, df_Tech_Skills, df_Interests]
df_names = ['df_Abilities', 'df_Knowledge', 'df_Skills', 'df_Tech_Skills', 'df_Interests']
for idx, i in enumerate(datasets):
    df_name = df_names[idx]
    field_name = df_name.split('_')[1]  # Extract category field from dataframe name
    key_col = ['O*NET-SOC Code', 'Description_' + field_name]
    df_job = pd.merge(df_job, i[key_col], how='left', on='O*NET-SOC Code')

df_job = df_job.fillna('')  # concatenate of NaN value with strings results in a NaN
df_job['Description_Job'] = df_job['Title'] + ' ' +df_job['Description'] + ' ' \
                              + df_job['Description_Abilities'] + ' ' + df_job['Description_Knowledge'] \
                              + ' ' + df_job['Description_Skills'] + ' ' + df_job['Description_Tech'] \
                              + ' ' + df_job['Description_Interests']

# Add job zone column to Job data
df_Job_Zone = read('Job Zones.txt')
#print(df_job.shape)
df_job = pd.merge(df_job, df_Job_Zone[['O*NET-SOC Code', 'Job Zone']], how='left', on='O*NET-SOC Code')
#print(df_job.shape)
df_job[['Description_Job','Job Zone']].head(3)
print('Finished every process of Creating Job Corpus\n')

#%%
# save_as_pickle(df_job, os.getcwd(), 'df_Occupation.pkl')  # Check Point
# df_job.to_csv(os.path.join(os.getcwd(), 'df_Occupation.csv'))
#%%
# init_chk_df_2(df_job)
# print(df_job.columns.to_list())



