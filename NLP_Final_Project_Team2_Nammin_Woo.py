#%%
# check path
import os
print(os.getcwd())
#%%
# Cloud directory (Nammin)
# path = '/home/ubuntu/Project'
# os.chdir(path)
# print(os.getcwd())

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
# C. Check individual files (.sql files)
path = r'./db_28_0_text' # Move to the file directory
os.chdir(path)
print(os.getcwd())
#%%
files = os.listdir('.')
onet_raw = [file for file in files if file.endswith('.txt')]
#%%
# 40 individual files
print(len(onet_raw)-1)  #Include ReadMe file
#%%
onet_raw.sort()
#%%
onet_raw
# for f in onet_raw:
#     print(f)
#%%
# [ Base: key, codes and description) ]
# A. O*NET-SOC Code: Occupation Data.txt,  Related Occupations.txt, Occupation Level Metadata.txt
# B. Element ID description: Content Model Reference.txt,
# *Sub: IWA Reference.txt (Intermediate Work Activity), DWA Reference.txt (Detailed)
# C. Scale ID: Scales Reference.txt
# D. Job Zone:  Job Zone Reference.txt
# E. Categories: Work Context Categories.txt, Task Categories,
#    Education, Training, and Experience Categories.txt
# F. UNSPSC taxonomy: UNSPSC Reference.txt (Family-Class-Comodity)

# [ Contents ]
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
df = read('Occupation Data.txt')
#%%
init_chk_df('Occupation Data.txt')
#%%
# A. Skimming ############################################
# Draft: Define skill-related files
skill_keywords =['Occupation Data.txt','Abilities.txt','Interests.txt','Knowledge.txt','Skills.txt'
    ,'Technology Skills.txt','Tools Used.txt','Work Activities.txt','Work Context.txt'
    , 'Work Styles.txt','Work Values.txt']
#%%
for file in skill_keywords:
    init_chk_df(file)
#%%
# B. Build up ############################################
#%%
'''
# Base data- #1. Occupation Data: Key-Title O*NET-SOC Code:
  - 1016 unique jobs with code, title, description'''

print(len(read('Occupation Data.txt')['O*NET-SOC Code'].unique()))
#%%
read('Occupation Data.txt').columns.to_list()

#%%
'''
# Contents data - #1. Abilities: Key-Title O*NET-SOC Code:

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
#init_chk_df('Abilities.txt')
# df = pd.read_csv('./Abilities.txt', sep='\t')  #90792 records
#%%
df.columns.to_list()
#%%
df.head()
#%%
# 873 unique occupations
print(len(df['O*NET-SOC Code'].unique()))
#%%
# Join Occupation name and description
print(df.shape)
key = 'O*NET-SOC Code'
df = pd.merge(df, read('Occupation Data.txt'), how='left', on= key)
print(df.shape)
#%%
df.head()
#%%
# 52 unique Ability names
print(df['Element Name'].unique())
print(len(df['Element Name'].unique()))
#%%
'''
# Base data- #2. Content Model Reference: Key-Element ID:
  - 627 unique Element ID with code, Name, description'''
# Base data- No 2 - Occupation file: Key-Title O*NET-SOC Code:
print(read('Content Model Reference.txt').columns.to_list())
print(len(read('Content Model Reference.txt')['Element ID'].unique()))
#%%
# Join Element description
print(df.shape)
key = 'Element ID'
join = ['Element ID', 'Description']
df = pd.merge(df, read('Content Model Reference.txt')[join], how='left', on= key)
print(df.shape)
#%%
# 2 unique values: IM: Importance, LV: Level
print(len(df['Scale ID'].unique()))
#%%
df['Scale ID'].value_counts(normalize = True).sort_index()  #50%,50%
#%%
'''
# Base data- #3. Scales Reference: Key-Scale ID:
  - 29 unique Scale ID with code, Name, Minimum, Maximum'''
print(read('Scales Reference.txt').columns.to_list())
print(len(read('Scales Reference.txt')['Scale ID'].unique()))

#%%
# Join Scale description
print(df.shape)
key = 'Scale ID'
df = pd.merge(df, read('Scales Reference.txt'), how='left', on= key)
print(df.shape)
#%%
# Function: make ratio variable: Relative place of the value
def value_ratio(df):
    df['Value_ratio'] = df['Data Value']/df['Maximum']
    return df
#%%
print(df.shape)
value_ratio(df)
print(df.shape)
#%%
#%%
# Updated date
df.Date.value_counts(normalize = True).sort_index()
#%%
# Analyst,Analyst - Transition
df['Domain Source'].value_counts(normalize = True).sort_index()
#%%
df.head()
#%%
df.columns.to_list()
#%%
df.rename(columns={'Description_x': 'Description_SOC', 'Description_y': 'Description_Ele'},inplace=True)
#%%
# (Final) PK check
pk = ['O*NET-SOC Code', 'Element ID', 'Scale ID']
temp = df.drop_duplicates(subset=pk)
print(df.shape[0], temp.shape[0])

#%%
# Sample check: occupation
keyword = 'statistics'
chk = [df.iloc[i]['Title'] for i in range(df.shape[0])
       if df.iloc[i]['Description_SOC'].lower().find(keyword) !=
       -1]
#%%
print(set(chk)) # 5 occupations containing 'statistics' keyword in descriptions
print(len(set(chk)))
#%%
filter = (df['Title'] == 'Statisticians') & (df['Element Name'] == 'Problem Sensitivity')
sample = df[filter]
#%%
col = ['Title', 'Element Name', 'Scale Name', 'Minimum', 'Maximum', 'Data Value', 'Value_ratio']
print(sample[col].to_string())
#%%
col = ['Title','Description_SOC']
filter = df['Scale ID'] == 'IM'
print(sample[col][filter].to_string())
#%%
col = ['Element Name', 'Description_Ele']
filter = df['Scale ID'] == 'IM'
print(sample[col][filter].to_string())

#%%
# Temporary : Save
import pickle
# df.to_pickle(("./df_Abilities.pkl"))
# df_Abilities = pd.read_pickle(("./df_Abilities.pkl"))
#%%
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
#init_chk_df('Interests.txt')
# df = pd.read_csv('./Interests.txt', sep='\t')  #7866 records
#%%
df.columns.to_list()
#%%
df.head() #
#%%
# 874 unique occupations
print(len(df['O*NET-SOC Code'].unique()))
#%%
print(len(df['Scale ID'].unique()))

#%%
def join_key(df):
    print(df.shape)
    df = pd.merge(df, read('Occupation Data.txt'), how='left', on='O*NET-SOC Code')
    df = pd.merge(df, read('Content Model Reference.txt')[['Element ID', 'Description']], how='left', on= 'Element ID')
    df = pd.merge(df, read('Scales Reference.txt'), how='left', on='Scale ID')
    df.rename(columns={'Description_x': 'Description_SOC', 'Description_y': 'Description_Ele'}, inplace=True)
    value_ratio(df)
    print(df.shape)
    return df
#%%
# Join Occupation name and description
df = join_key(df)
#%%
df.columns.to_list()
#%%
df['Scale Name'].value_counts(normalize = True).sort_index()  #67%,33%
#%%
print(df['Element Name'].unique()) # 10 interests
#%%
# (Final) PK check
pk = ['O*NET-SOC Code', 'Element ID', 'Scale ID']
temp = df.drop_duplicates(subset=pk)
print(df.shape[0], temp.shape[0])
#%%
# Sample check: occupation
filter = (df['Title'] == 'Statisticians') & (df['Element Name'] == 'Investigative')
sample = df[filter]
#%%
col = ['Title', 'Element Name', 'Scale Name', 'Minimum', 'Maximum', 'Data Value', 'Value_ratio']
print(sample[col].to_string())
#%%
filter = (df['Title'] == 'Statisticians') & (df['Scale Name'] == 'Occupational Interests')
sample = df[filter]
print(sample[col].to_string())
#%%
filter = (df['Title'] == 'Statisticians') & (df['Scale Name'] == 'Occupational Interest High-Point')
sample = df[filter]
print(sample[col].to_string())
#%%
# Temporary : Save
#df.to_pickle(("./df_Interests.pkl"))
# df_Abilities = pd.read_pickle(("./df_Interests.pkl"))

#%%
# print(os.getcwd())
# path = r'/home/ubuntu/Project/db_28_0_text' # Move to the file directory
# os.chdir(path)
# print(os.getcwd())