#  Section 1. Read O*NET® 28.0 Database on the WEB

###  1. Get Raw data (40 individual files were ziped) was downloaded on the Web download link 

      - Zip file was called using request and extracted into individual *.txt files
      - Among 40 files, followings contains BASE information (code-description mapping) 
        A. O*NET-SOC Code: Occupation Data.txt,  Related Occupations.txt, Occupation Level Metadata.txt,
        B. Element ID description: Content Model Reference.txt,
           *Sub: IWA Reference.txt (Intermediate Work Activity), DWA Reference.txt (Detailed)
        C. Scale ID: Scales Reference.txt
        D. Job Zone: Job Zone Reference.txt
        E. Categories: Work Context Categories.txt, Task Categories, Education, Training, and Experience Categories.txt
        F. UNSPSC taxonomy: UNSPSC Reference.txt (Family-Class-Comodity)

      - Others contains CONTENTS related with each infomation field above
        ex. 'Abilities.txt',..Job Zones.txt, ..,'Knowledge.txt'

#### Reference information
 -  [Current released version O*NET database](https://www.onetcenter.org/database.html)
 -  [Basic- Occupation DB](https://www.onetcenter.org/dictionary/28.0/mysql/occupation_data.html)
 -  [All Contents of the Data](https://www.onetcenter.org/dictionary/28.0/text/)


###  2. Basic exploration and Preprocessing 

      - Among 40 files, 10 files about individual characteristics are choosed
       'Abilities.txt','Interests.txt','Knowledge.txt'
       'Skills.txt','Technology Skills.txt','Tools Used.txt'
       'Work Activities.txt','Work Context.txt', 'Work Styles.txt','Work Values.txt'

      - 10 files converted into DataFrame after aggregating with BASE files
        ex.  1. Join Occupation name and description from Occupation Data
             2. Join Element description from Content Model Reference
             3. Join Scale description (Minimum, Maximum) from Content Scales Reference

      - 5 DataFrames merged into 'df_Occupation' with final preprocessing (Make JOB Corpus) 
         'Abilities.txt','Interests.txt','Knowledge.txt', 'Skills.txt','Technology Skills.txt'

###  3. Make Job Description Corpus (for 1000 Occupations in ONET)
       1. Define 5 ONET data Categories to Construct Corpus
         df_names = ['df_Abilities', 'df_Knowledge', 'df_Skills','df_Tech_Skills', 'df_Interests' ]
         *Exclude: 'df_Tools_Used', 'df_Work_Values', 'df_Work_Activities', 'df_Work_Styles','df_Work_Context'
        
       2. Make Job_Description corpus of each category datasets 
       *Rule (pay attention to Key elements in an Occupation)
       Step 0. Start from the first Category (ex. Ability)
       step 1. Choose High Demand Elements in an Occupation (Level Demand ratio over 50%, Pick Top 3)
       step 2. Create Description column by an Occupation: (ex. Description_Abilities)
       'Category: + 'Descriptions of High Demand elements
       (ex) Abilities: The ability to A ...  The ability to B ... The ability to C ...
       Step 3. Iterate for of 5 Categories : Get 5 Description columns
       Description_Abilities, ... , Description_Interests
    
       3. Make Final Job dataset with Full_Job_Description corpus (Merge of 5 Job_Description corpus)
       Join to the Job Occupation dataset
       'Job_Description' + 'Description_Abilities' .. + 'Description_Interests'

#### Appendix 
     - Outline of individual files 

     [ Base: key, codes and description) ]: Unfold below to see details
    
     A. O*NET-SOC Code: Occupation Data.txt,  Related Occupations.txt, Occupation Level Metadata.txt
     B. Element ID description: Content Model Reference.txt,
     *Sub: IWA Reference.txt (Intermediate Work Activity), DWA Reference.txt (Detailed)
     C. Scale ID: Scales Reference.txt
     D. Job Zone:  Job Zone Reference.txt
     E. Categories: Work Context Categories.txt, Task Categories,
        Education, Training, and Experience Categories.txt
     F. UNSPSC taxonomy: UNSPSC Reference.txt (Family-Class-Comodity)
      
    
     [ Contents ] : Unfold below to see details
    
     'Abilities to Work Activities.txt', 'Abilities to Work Context.txt', 'Abilities.txt',
      'Alternate Titles.txt', 'Basic Interests to RIASEC.txt',
      'Education, Training, and Experience.txt', 'Emerging Tasks.txt',
      'Interests Illustrative Activities.txt', 'Interests Illustrative Occupations.txt',
      'Interests.txt', 'Job Zones.txt,
      'Knowledge.txt', 'Level Scale Anchors.txt',
      'RIASEC Keywords.txt', 'Related Occupations.txt', 'Sample of Reported Titles.txt',
      'Scales Reference.txt', 'Skills to Work Activities.txt', 'Skills to Work Context.txt',
      'Skills.txt', 'Survey Booklet Locations.txt',
      'Task Ratings.txt', 'Task Statements.txt', 'Tasks to DWAs.txt',
      'Technology Skills.txt', 'Tools Used.txt'
      'Work Activities.txt', 'Work Context.txt', 'Work Styles.txt', 'Work Values.txt']


        - ONET(JOB)
        Shape of df_Knowledge: (57618, 20)
        Shape of df_Work_Values: (7866, 14)
        Shape of df_Skills: (61110, 20)
        Shape of df_Work_Activities: (71586, 20)
        Shape of df_Work_Styles: (13968, 19)
        Shape of df_Interests: (7866, 14)
        Shape of df_Abilities: (90792, 20)
        Shape of df_Tools_Used: (41644, 6)
        Shape of df_Tech_Skills: (32384, 8)
        Shape of df_Work_Context: (289173, 22)

        - Resume
        Shape of resume_data_cleaned: (962, 2)


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

#%%
''' #9. Work Styles: Title-Element-Scale'''
#%%  : Unfold below to see details
'''
# Contents data(Sample) :
        Title Element Name  Scale Name  Minimum  Maximum  Data Value  Value_ratio
1873  Statisticians  Persistence  Importance        1        5         4.0          0.8
'''

#%%
''' #10. Work Values: Title-Element-Scale'''
#%%  : Unfold below to see details
'''
# Contents data(Sample) :
            Title   Element Name Scale Name  Minimum  Maximum  Data Value  Value_ratio
1065  Statisticians  Relationships     Extent        1        7        4.67     0.667143
'''