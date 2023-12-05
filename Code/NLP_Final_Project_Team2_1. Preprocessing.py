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
    except Exception as e:
        print(f"Error creating 'Project' directory: {e}")
else:
    print(f"'Project' directory already exists: {project_directory}")

# Now your current working directory should be /home/ubuntu/Project
print(f"Current working directory: {os.getcwd()}")



#%%
###############################################################################
# Section 1. Load Rawdata:  O*NET® 28.0 Database on the WEB
###############################################################################

#%%
import pandas as pd
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
#%%
# A. Define Download Link: Choose txt format (sql: create sql database)
url = r'https://www.onetcenter.org/dl_files/database/db_28_0_text.zip'
#%%
# B. Read Bulk data (*.zip file) and unzip to individual files
def open_url_zip(url):
    zip = urlopen(url)  #request.urlopen(url)
    # Extract zip file
    with ZipFile(BytesIO(zip.read())) as zfile:
        zfile.extractall()
    return
#%%
open_url_zip(url)

#%%
# Move to the file directory
path = r'/home/ubuntu/Project/db_28_0_text'
os.chdir(path)
print(os.getcwd())

#%%

# Set the desired working directory path
new_directory = "/home/ubuntu/Project/db_28_0_text/"

# Change the current working directory
os.chdir(new_directory)

# Verify the new working directory
print(f"Changed working directory to: {os.getcwd()}")


#%%
import os

# Get the current working directory
current_directory = os.getcwd()

# List files in the current working directory
files = os.listdir(current_directory)

# Filter for files ending with '.txt'
onet_raw = [file for file in files if file.endswith('.txt')]

# Print the list of matching files
print(onet_raw)

#%%
# 40 individual files
print(len(onet_raw)-1)  #Include ReadMe file
#%%
onet_raw.sort()
#%%
# for f in onet_raw:print(f)

# Outline of individual files #################################################
#
# [ Base: key, codes and description) ]: Unfold below to see details
#%%
# A. O*NET-SOC Code: Occupation Data.txt,  Related Occupations.txt, Occupation Level Metadata.txt
# B. Element ID description: Content Model Reference.txt,
# *Sub: IWA Reference.txt (Intermediate Work Activity), DWA Reference.txt (Detailed)
# C. Scale ID: Scales Reference.txt
# D. Job Zone:  Job Zone Reference.txt
# E. Categories: Work Context Categories.txt, Task Categories,
#    Education, Training, and Experience Categories.txt
# F. UNSPSC taxonomy: UNSPSC Reference.txt (Family-Class-Comodity)
  #
#%%
# [ Contents ] : Unfold below to see details
#%%
# 'Abilities to Work Activities.txt', 'Abilities to Work Context.txt', 'Abilities.txt',
#  'Alternate Titles.txt', 'Basic Interests to RIASEC.txt',
#  'Education, Training, and Experience.txt', 'Emerging Tasks.txt',
#  'Interests Illustrative Activities.txt', 'Interests Illustrative Occupations.txt',
#  'Interests.txt', 'Job Zones.txt,
#  'Knowledge.txt', 'Level Scale Anchors.txt',
#  'RIASEC Keywords.txt', 'Related Occupations.txt', 'Sample of Reported Titles.txt',
#  'Scales Reference.txt', 'Skills to Work Activities.txt', 'Skills to Work Context.txt',
#  'Skills.txt', 'Survey Booklet Locations.txt',
#  'Task Ratings.txt', 'Task Statements.txt', 'Tasks to DWAs.txt',
#  'Technology Skills.txt', 'Tools Used.txt'
#  'Work Activities.txt', 'Work Context.txt', 'Work Styles.txt', 'Work Values.txt']

#%%
###############################################################################
# Section 2. Basic Analysis:
###############################################################################
#%%
# Class define (TBD)
# class ONET:
#     def __init__(self,col):
#         self.col = self
#
#     def init_chk_df(self):

#%%
# Function: Read Individual file and show basic information
def read(file):
    path = os.path.join(os.getcwd(), file)
    df = pd.read_csv(path, sep='\t')
    return df

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
#%%
'''# Base data- #1. Occupation Data:
  - 1016 unique jobs with code, title, description'''
#%%
init_chk_df('Occupation Data.txt')
df = read('Occupation Data.txt')
print(len(df['O*NET-SOC Code'].unique()))

# keyword = 'statistics'
# chk = [df.iloc[i]['Title'] for i in range(df.shape[0])
#        if df.iloc[i]['Description'].lower().find(keyword) !=
#        -1]
# print(set(chk))  # 5 occupations containing 'statistics' keyword in descriptions

#%%
# A. Skimming data files ############################################
# Draft: Define skill-related files
skill_keywords =['Abilities.txt','Interests.txt','Knowledge.txt','Skills.txt'
    ,'Technology Skills.txt','Tools Used.txt','Work Activities.txt','Work Context.txt'
    , 'Work Styles.txt','Work Values.txt']
#%%
for file in skill_keywords:
    init_chk_df(file)
#%%
# B. Aggregate dataset ############################################
#%%
# Function: make ratio variable: Relative place of the value
def value_ratio(df):
    df['Value_ratio'] = df['Data Value']/df['Maximum']
    return df

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

#%%
''' #1. Abilities: Title-Element-Scale'''
#%%  : Unfold below to see details
'''
['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Data Value',
  ...'Date', 'Domain Source']
  
- PK: SOC Code (873 Occupations) - Element (52 Abilities) - Scale ID (2 Types of value)
- 90792 records (873 * 52 * 2)
- Two Text descriptions: Description_SOC(Job),  Description_Ele(Abilities)
- Created 'Value_ratio' : Data Value/Maximum
- Example 
: *** idea 
: Problem Sensitivity ability in Statisticians occupation need Level, have importance
   
               Title         Element Name  Scale Name  Minimum  Maximum  Data Value  Value_ratio
12180  Statisticians  Problem Sensitivity  Importance        1        5        3.38     0.676000
12181  Statisticians  Problem Sensitivity       Level        0        7        3.75     0.535714
  
'''

#%%
df = read('Abilities.txt')
# Aggregate tables (Content dataframe + Base tables)
df = join_key(df)
df.head()
# df.columns.to_list()
#%%
# Check diversity
print(len(df['O*NET-SOC Code'].unique())) # 873 unique occupations
print(len(df['Scale ID'].unique()), '\n') # 2 models
print(df['Scale Name'].value_counts(normalize = True).sort_index(),'\n')  #IM, LV
print(len(df['Element Name'].unique())) # 52 Abilities
print(df['Element Name'].unique())

# # Updated date
# df.Date.value_counts(normalize = True).sort_index()
# # Analyst,Analyst - Transition
# df['Domain Source'].value_counts(normalize = True).sort_index()
# Verify PK of dataset
# pk = ['O*NET-SOC Code', 'Element ID', 'Scale ID']
# temp = df.drop_duplicates(subset=pk)
# print(df.shape[0], temp.shape[0])

#%% Sample check: Unfold below to see details
#%%
filter = (df['Title'] == 'Statisticians') & (df['Element Name'] == 'Problem Sensitivity')
sample = df[filter]

col = ['Title', 'Element Name', 'Scale Name', 'Minimum', 'Maximum', 'Data Value', 'Value_ratio']
print(sample[col].to_string())

col = ['Title','Description_SOC']
filter = df['Scale ID'] == 'IM'
print(sample[col][filter].to_string())

col = ['Element Name', 'Description_Ele']
filter = df['Scale ID'] == 'IM'
print(sample[col][filter].to_string())
#%%
df_Abilities = df
#%%
''' #2. Interests: Title-Element-Scale'''
#%%  : Unfold below to see details
'''
# Contents data - #2. Interests: Key-Title O*NET-SOC Code:

              Title   Element Name              Scale Name  Minimum  Maximum  Data Value  Value_ratio
1062  Statisticians      Realistic  Occupational Interests        1        7        2.33     0.332857
1063  Statisticians  Investigative  Occupational Interests        1        7        6.00     0.857143
1064  Statisticians       Artistic  Occupational Interests        1        7        2.00     0.285714
1065  Statisticians         Social  Occupational Interests        1        7        1.00     0.142857
1066  Statisticians   Enterprising  Occupational Interests        1        7        2.00     0.285714
1067  Statisticians   Conventional  Occupational Interests        1        7        6.33     0.904286
'''
#%%
df = read('Interests.txt')
# Aggregate tables (Content dataframe + Base tables)
df = join_key(df)
#%%
# Check diversity
print(len(df['O*NET-SOC Code'].unique())) # 874 unique occupations
print(len(df['Scale ID'].unique()), '\n')
print(df['Scale Name'].value_counts(normalize = True).sort_index(),'\n')  #67%,33%
print(len(df['Element Name'].unique())) # 10 interests
print(df['Element Name'].unique())

#%% Sample check: Unfold below to see details

filter = (df['Title'] == 'Statisticians') & (df['Element Name'] == 'Investigative')
sample = df[filter]
col = ['Title', 'Element Name', 'Scale Name', 'Minimum', 'Maximum', 'Data Value', 'Value_ratio']
print(sample[col].to_string())

filter = (df['Title'] == 'Statisticians') & (df['Scale Name'] == 'Occupational Interests')
# filter = (df['Title'] == 'Statisticians') & (df['Scale Name'] == 'Occupational Interest High-Point')
sample = df[filter]
print(sample[col].to_string())
#%%
df_Interests = df

#%%
''' #3. Knowledge: Title-Element-Scale'''
#%%  : Unfold below to see details
'''
# Contents data(Sample) :
              Title    Element Name  Scale Name  Minimum  Maximum  Data Value  Value_ratio
7724  Statisticians  Administrative  Importance        1        5        1.72     0.344000
7725  Statisticians  Administrative       Level        0        7        1.94     0.277143
              Title                   Element Name  Scale Name  Minimum  Maximum  Data Value  Value_ratio
7722  Statisticians  Administration and Management  Importance        1        5        2.21        0.442
7724  Statisticians                 Administrative  Importance        1        5        1.72        0.344
7726  Statisticians       Economics and Accounting  Importance        1        5        1.78        0.356
7728  Statisticians            Sales and Marketing  Importance        1        5        1.44        0.288
'''
#%%
df = read('Knowledge.txt')
print(df.head())
#%%
# Aggregate tables (Content dataframe + Base tables)
df = join_key(df)
#%%
# Check diversity
print(len(df['O*NET-SOC Code'].unique())) # 873 unique occupations
print(len(df['Scale ID'].unique()), '\n') # 2
print(df['Scale Name'].value_counts(normalize = True).sort_index(),'\n')  #IM, LV
print(len(df['Element Name'].unique())) # 33 elements
print(df['Element Name'].unique())

#%% Sample check: Unfold below to see details

filter = (df['Title'] == 'Statisticians') & (df['Element Name'] == 'Administrative')
sample = df[filter]
col = ['Title', 'Element Name', 'Scale Name', 'Minimum', 'Maximum', 'Data Value', 'Value_ratio']
print(sample[col].to_string())
print('\n')
filter = (df['Title'] == 'Statisticians') & (df['Scale Name'] == 'Importance')
sample = df[filter]
print(sample[col].to_string())
#%%
df_Knowledge = df
#%%
''' #4. Skills: Title-Element-Scale'''
#%%  : Unfold below to see details
'''
# Contents data(Sample) :
              Title Element Name  Scale Name  Minimum  Maximum  Data Value  Value_ratio
8194  Statisticians      Writing  Importance        1        5        3.50     0.700000
8195  Statisticians      Writing       Level        0        7        4.38     0.625714
              Title                       Element Name  Scale Name  Minimum  Maximum  Data Value  Value_ratio
8190  Statisticians              Reading Comprehension  Importance        1        5        4.00        0.800
8192  Statisticians                   Active Listening  Importance        1        5        3.88        0.776
8194  Statisticians                            Writing  Importance        1        5        3.50        0.700
'''
#%%
df = read('Skills.txt')
#print(df.head())
#%%
# Aggregate tables (Content dataframe + Base tables)
df = join_key(df)
#%%
# Check diversity
print(len(df['O*NET-SOC Code'].unique())) # 873 unique occupations
print(len(df['Scale ID'].unique()), '\n') # 2
print(df['Scale Name'].value_counts(normalize = True).sort_index(),'\n')  #IM, LV
print(len(df['Element Name'].unique())) # 35 elements
print(df['Element Name'].unique())

#%% Sample check: Unfold below to see details

filter = (df['Title'] == 'Statisticians') & (df['Element Name'] == 'Writing')
sample = df[filter]
col = ['Title', 'Element Name', 'Scale Name', 'Minimum', 'Maximum', 'Data Value', 'Value_ratio']
print(sample[col].to_string())
print('\n')
filter = (df['Title'] == 'Statisticians') & (df['Scale Name'] == 'Importance')
sample = df[filter]
print(sample[col].to_string())
#%%
df_Skills = df
#%%
''' #5. Technology Skills: Title-Example (No element and Scale)'''
#%%  : Unfold below to see details
'''
# Contents data(Sample) :
 O*NET-SOC Code          Title                                      Example  Commodity Code                                    Commodity Title Hot Technology In Demand
11711     15-2041.00  Statisticians                              Amazon Redshift        43232306        Data base user interface and query software              Y         N
11713     15-2041.00  Statisticians                                Apache Hadoop        43232304               Data base management system software              Y         N

      O*NET-SOC Code          Title                         Example  Commodity Code                                    Commodity Title Hot Technology In Demand
11724     15-2041.00  Statisticians  Extensible markup language XML        43232403        Enterprise application integration software              Y         Y
11734     15-2041.00  Statisticians                 Microsoft Excel        43232110                               Spreadsheet software              Y         Y
'''
#%%
df = read('Technology Skills.txt')
print(df.head(1).to_string())

#%%
# Aggregate tables (Only + Job description)
df = pd.merge(df, read('Occupation Data.txt'), how='left', on='O*NET-SOC Code')
#%%
# Check diversity
print(len(df['O*NET-SOC Code'].unique())) # 923 unique occupations
print(len(df['Commodity Code'].unique()), '\n') # 135
print(df['Commodity Title'].unique())
#%%
# pk = ['O*NET-SOC Code', 'Example']
# temp = df.drop_duplicates(subset=pk)
# print(df.shape[0], temp.sha
#%% Sample check: Unfold below to see details
col2 = ['O*NET-SOC Code','Title', 'Example', 'Commodity Code', 'Commodity Title', 'Hot Technology', 'In Demand']
filter = (df['Title'] == 'Statisticians') & (df['Hot Technology'] == 'Y')
sample = df[filter]
print(sample[col2].to_string())
print('\n')
filter = (df['Title'] == 'Statisticians') & (df['In Demand'] == 'Y')
sample = df[filter]
print(sample[col2].to_string())
#%%

df_Tech_Skills = df
#%%
''' #6. Tools Used: Title-Example (No element and Scale)'''
#%%  : Unfold below to see details
'''
# Contents data(Sample) :
 O*NET-SOC Code          Title             Example  Commodity Code     Commodity Title
1697     15-2041.00  Statisticians   Desktop computers        43211507   Desktop computers
1698     15-2041.00  Statisticians    Laptop computers        43211503  Notebook computers
1699     15-2041.00  Statisticians  Personal computers        43211508  Personal computers
'''
#%%
df = read('Tools Used.txt')
print(df.head(1).to_string())
#%%
# Aggregate tables (Only + Job description)
df = pd.merge(df, read('Occupation Data.txt'), how='left', on='O*NET-SOC Code')
#%%
# Check diversity
print(len(df['O*NET-SOC Code'].unique())) # 902 unique occupations
print(len(df['Commodity Code'].unique()), '\n') # 4127
print(df['Commodity Title'].unique())
#%% Sample check: Unfold below to see details
col2 = ['O*NET-SOC Code','Title', 'Example', 'Commodity Code', 'Commodity Title']
filter = (df['Title'] == 'Statisticians')
sample = df[filter]
print(sample[col2].to_string())
#%%
df_Tools_Used = df
#%%
''' #7. Work Activities: Title-Element-Scale'''
#%%  : Unfold below to see details
'''
# Contents data(Sample) :
              Title         Element Name  Scale Name  Minimum  Maximum  Data Value  Value_ratio
9594  Statisticians  Getting Information  Importance        1        5         4.5     0.900000
9595  Statisticians  Getting Information       Level        0        7         5.2     0.742857
              Title                                                                     Element Name  Scale Name  Minimum  Maximum  Data Value  Value_ratio
9594  Statisticians                                                              Getting Information  Importance        1        5        4.50        0.900
9596  Statisticians                                 Monitoring Processes, Materials, or Surroundings  Importance        1        5        2.25        0.450
9598  Statisticians                                         Identifying Objects, Actions, and Events  Importance        1        5        3.40        0.680
'''
#%%
df = read('Work Activities.txt')
print(df.head(1).to_string())
#%%
# Aggregate tables (Content dataframe + Base tables)
df = join_key(df)
#%%
# Check diversity
print(len(df['O*NET-SOC Code'].unique())) # 873 unique occupations
print(len(df['Scale ID'].unique()), '\n') # 2
print(df['Scale Name'].value_counts(normalize = True).sort_index(),'\n')  #IM, LV
print(len(df['Element Name'].unique())) # 41 elements
print(df['Element Name'].unique())

#%% Sample check: Unfold below to see details

filter = (df['Title'] == 'Statisticians') & (df['Element Name'] == 'Getting Information')
sample = df[filter]
col = ['Title', 'Element Name', 'Scale Name', 'Minimum', 'Maximum', 'Data Value', 'Value_ratio']
print(sample[col].to_string())
print('\n')
filter = (df['Title'] == 'Statisticians') & (df['Scale Name'] == 'Importance')
sample = df[filter]
print(sample[col].to_string())

#%%
df_Work_Activities = df

#%%
''' #8. Work Context: Title-Element-Scale-Category(Nan, Category)'''
#%%  : Unfold below to see details
'''
# Contents data(Sample) :
               Title              Element Name                Scale Name  Minimum  Maximum  Category                     Category Description  Data Value  Value_ratio
37884  Statisticians  Face-to-Face Discussions                   Context        1        5       NaN                                      NaN         4.4         0.88
37885  Statisticians  Face-to-Face Discussions  Context (Categories 1-5)        0      100       1.0                                    Never         0.0         0.00
37886  Statisticians  Face-to-Face Discussions  Context (Categories 1-5)        0      100       2.0  Once a year or more but not every month         0.0         0.00
37887  Statisticians  Face-to-Face Discussions  Context (Categories 1-5)        0      100       3.0  Once a month or more but not every week         0.0         0.00
37888  Statisticians  Face-to-Face Discussions  Context (Categories 1-5)        0      100       4.0    Once a week or more but not every day        60.0         0.60
37889  Statisticians  Face-to-Face Discussions  Context (Categories 1-5)        0      100       5.0                                Every day        40.0         0.40
'''
#%%
df = read('Work Context.txt')
print(df.head(1).to_string())
#%%
# Aggregate tables (Content dataframe + Base tables)
df = join_key(df)
#%%
# Aggregate again (+ Job description)
df = pd.merge(df, read('Work Context Categories.txt')[['Element ID','Category', 'Category Description']]
              , how='left', on=['Element ID','Category'])
#%%
# Check diversity
print(len(df['O*NET-SOC Code'].unique())) # 873 unique occupations
print(len(df['Scale ID'].unique()), '\n') # 4
print(df['Scale Name'].value_counts(normalize = True).sort_index(),'\n')  #IM, LV
print(len(df['Element Name'].unique())) # 57 elements
print(len(df['Category'].unique())) # 6 elements
print(df['Category'].unique()) # [nan  1.  2.  3.  4.  5.]
#%% Sample check: Unfold below to see details
col2 = ['Title', 'Element Name', 'Scale Name', 'Minimum', 'Maximum','Category','Category Description', 'Data Value',
        'Value_ratio']
filter = (df['Title'] == 'Statisticians') & (df['Element Name'] == 'Face-to-Face Discussions')
sample = df[filter]
print(sample[col2].to_string())
print('\n')
filter = (df['Title'] == 'Statisticians') & (df['Element Name'] == 'Face-to-Face Discussions')& (df['Scale Name'] == 'Context')
sample = df[filter]
print(sample[col2].to_string())
#%%
df_Work_Context = df

#%%
''' #9. Work Styles: Title-Element-Scale'''
#%%  : Unfold below to see details
'''
# Contents data(Sample) :
        Title Element Name  Scale Name  Minimum  Maximum  Data Value  Value_ratio
1873  Statisticians  Persistence  Importance        1        5         4.0          0.8
'''
#%%
df = read('Work Styles.txt')
print(df.head(1).to_string())
#%%
# Aggregate tables (Content dataframe + Base tables)
df = join_key(df)
#%%
# Check diversity
print(len(df['O*NET-SOC Code'].unique())) # 873 unique occupations
print(len(df['Scale ID'].unique()), '\n') # 1
print(df['Scale Name'].value_counts(normalize = True).sort_index(),'\n')  #IM
print(len(df['Element Name'].unique())) # 16 elements
print(df['Element Name'].unique())
#%% Sample check: Unfold below to see details
col2 = ['Title', 'Element Name', 'Scale Name', 'Minimum', 'Maximum','Data Value','Value_ratio']
filter = (df['Title'] == 'Statisticians') & (df['Element Name'] == 'Persistence')
sample = df[filter]
print(sample[col2].to_string())
#%%
df_Work_Styles = df

#%%
''' #10. Work Values: Title-Element-Scale'''
#%%  : Unfold below to see details
'''
# Contents data(Sample) :
            Title   Element Name Scale Name  Minimum  Maximum  Data Value  Value_ratio
1065  Statisticians  Relationships     Extent        1        7        4.67     0.667143
'''
#%%
df = read('Work Values.txt')
print(df.head(1).to_string())
#%%
# Aggregate tables (Content dataframe + Base tables)
df = join_key(df)
#%%
# Check diversity
print(len(df['O*NET-SOC Code'].unique())) # 874 unique occupations
print(len(df['Scale ID'].unique()), '\n') # 2
print(df['Scale Name'].value_counts(normalize = True).sort_index(),'\n')  #Extent, Work Value High-Point
print(len(df['Element Name'].unique())) # 9 elements
print(df['Element Name'].unique())
#%% Sample check: Unfold below to see details
col2 = ['Title', 'Element Name', 'Scale Name', 'Minimum', 'Maximum','Data Value','Value_ratio']
filter = (df['Title'] == 'Statisticians') & (df['Element Name'] == 'Relationships')
sample = df[filter]
print(sample[col2].to_string())
#%%
df_Work_Values = df
#%%
# Check Point: Load Temporary Saved DataFrame

# Temporary : Save
# df.to_pickle(("./df_Work_Values.pkl"))
# df_Abilities = pd.read_pickle(("./df_Work_Values.pkl"))

#%%
print(os.getcwd())
#%%
# Load *.pkl dataset
for filename in os.listdir(os.getcwd()):
    # check if the file is a .pkl file
    if filename.endswith('.pkl'):
        filepath = os.path.join(os.getcwd(), filename)
        df_name = filename.split('.')[0]
        globals()[df_name] = pd.read_pickle(filepath)
        print(f'Shape of {df_name}: {globals()[df_name].shape}')
#%%

# Shape of df_Knowledge: (57618, 20)
# Shape of df_Work_Values: (7866, 14)
# Shape of df_Skills: (61110, 20)
# Shape of df_Work_Activities: (71586, 20)
# Shape of df_Work_Styles: (13968, 19)
# Shape of df_Interests: (7866, 14)
# Shape of df_Abilities: (90792, 20)
# Shape of df_Tools_Used: (41644, 6)
# Shape of df_Tech_Skills: (32384, 8)
# Shape of df_Work_Context: (289173, 22)

#%%

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
else:
    print(f"'{folder_name}' folder already exists inside '{project_directory}'")


#%%
# First download kaggle in mobaxtream termnal: pip install kaggle
# Then you will see kaggle "/home/ubuntu/.kaggle/"
# Then go to this: https://www.kaggle.com/settings
# Then press on "Create New Token" which is your auth keys in json file
# it will be downloaded in to your local
# move it Mobaextreme '.kaggle' and past it there
# so kaggle will work when you import it

import os
import os
import pandas as pd
import kaggle

def download_dataset(dataset_name, destination_folder):
    try:
        kaggle.api.dataset_download_files(dataset_name, path=destination_folder, unzip=True)
        print(f"Dataset downloaded to '{destination_folder}'")
        return True
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")
        return False

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
    dataset_name = 'gauravduttakiit/resume-dataset'
    destination_folder = '/home/ubuntu/Project/resume'  # Complete the path

    # Downloading the dataset
    if download_dataset(dataset_name, destination_folder):
        file_path = os.path.join(destination_folder, 'UpdatedResumeDataSet.csv')
        df_resume = read_dataset(file_path)
        if df_resume is not None:
            df_resume_clean = clean_resume_data(df_resume)
            initial_analysis(df_resume_clean)

if __name__ == "__main__":
    main()
#%%
    """
    # Initial Analysis of the Resume Dataset
    # First 5 rows:
    #       Category                                             Resume
    # 0  Data Science  Skills * Programming Languages: Python (pandas...
    # 1  Data Science  Education Details  May 2013 to May 2017 B.E   ...
    # 2  Data Science  Areas of Interest Deep Learning, Control Syste...
    # 3  Data Science  Skills - R - Python - SAP HANA - Tableau - SAP...
    # 4  Data Science  Education Details   MCA   YMCAUST,  Faridabad,...

    # Data types:
    # Category    object
    # Resume      object
    # dtype: object

    # Null values in each column:
    # Category    0
    # Resume      0
    # dtype: int64

    # Unique categories:
    # Java Developer               84
    # Testing                      70
    # DevOps Engineer              55
    # Python Developer             48
    # Web Designing                45
    # HR                           44
    # Hadoop                       42
    # Blockchain                   40
    # ETL Developer                40
    # Operations Manager           40
    # Data Science                 40
    # Sales                        40
    # Mechanical Engineer          40
    # Arts                         36
    # Database                     33
    # Electrical Engineering       30
    # Health and fitness           30
    # PMO                          30
    # Business Analyst             28
    # DotNet Developer             28
    # Automation Testing           26
    # Network Security Engineer    25
    # SAP Developer                24
    # Civil Engineer               24
    # Advocate                     20
    # dtype: int64
    """


#%%
# Saving the file to the Cloud on the cloened git repo
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
        destination_folder = '/home/ubuntu/job_search/Data_cleaned'
        file_name = 'resume_data_cleaned.pkl'
        print("Attempting to save DataFrame as pickle...")
        save_as_pickle(df_cleaned, destination_folder, file_name)
    else:
        print("Failed to load dataset.")


if __name__ == "__main__":
    main()

# Then I Draged to the local file Cleaned data...