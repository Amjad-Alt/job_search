#  Below table shows final codes in this project. 
     
    - Codes contain from creating data, training model, and Web application code. 
    - Number means order of running.
    - Three codes (#1,#2,#5) generate outputs that are used in other codes.

| Code                                         | Goal                                        | Output                              |
|----------------------------------------------|---------------------------------------------|-------------------------------------|
| Utils_Team2                                  | Gather user-defined functions                |                                     |
| 1.Create_Job_Corpus_Data_Team2               | Create Job corpus                           | df_Occupation.csv/pkl              |
| 2.Train_Model_Job_level_Classification_Team2 | Final model (Finetune_BERT)      | trained_model.pth                   |
| 3.logistic_zone                              | Train Classical model (Logistic Regression) |                                     |
| 4.MLP_zone                                   | Train Classical model (Naive Bayes)         |                                     |
| 5.Semantic_Similarity_Search_Team2           | Get embedding vector of Job corpus         | job_encodings.pkl                   |
| 6.Streamlit_script                           | Run Streamlit App                           |                                     |

##  Summary of Getting O*NET® 28.0 Database and Preprocessing it

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
       step 1. Choose High Demand Elements in an Occupation (Level Demand ratio over 50%, Pick Top 5)
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


# 1. Abilities: Title-Element-Scale

- PK: SOC Code (873 Occupations) - Element (52 Abilities) - Scale ID (2 Types of value)
  - 90792 records (873 * 52 * 2)
  - Two Text descriptions: Description_SOC(Job), Description_Ele(Abilities)
  - Created 'Value_ratio': Data Value/Maximum
  - Element: 
    - Name 
    - Description (from Base information table) 
    
    ex.Problem Sensitivity ability:The ability to tell when something is wrong or is likely to go wrong. 
    It does not involve solving the problem, only recognizing that there is a problem.
  
| Title         | Element Name         | Scale Name | Minimum | Maximum | Data Value | Value_ratio |
|---------------|----------------------|------------|---------|---------|------------|-------------|
| Statisticians | Problem Sensitivity  | Importance | 1       | 5       | 3.38       | 0.676       |
| Statisticians | Problem Sensitivity  | Level      | 0       | 7       | 3.75       | 0.536       |

# 2. Interests: Title-Element-Scale

[Contents data - #2. Interests: Key-Title O*NET-SOC Code]

| Title         | Element Name      | Scale Name               | Minimum | Maximum | Data Value | Value_ratio |
|---------------|-------------------|--------------------------|---------|---------|------------|-------------|
| Statisticians | Realistic         | Occupational Interests    | 1       | 7       | 2.33       | 0.333       |
| Statisticians | Investigative     | Occupational Interests    | 1       | 7       | 6.00       | 0.857       |
| ...           | ...               | ...                      | ...     | ...     | ...        | ...         |

# 3. Knowledge: Title-Element-Scale

[Contents data(Sample)]

| Title         | Element Name                   | Scale Name | Minimum | Maximum | Data Value | Value_ratio |
|---------------|--------------------------------|------------|---------|---------|------------|-------------|
| Statisticians | Administrative                 | Importance | 1       | 5       | 1.72       | 0.344       |
| Statisticians | Administration and Management  | Importance | 1       | 5       | 2.21       | 0.442       |
| ...           | ...                            | ...        | ...     | ...     | ...        | ...         |

# 4. Skills: Title-Element-Scale

[Contents data(Sample)]

| Title         | Element Name            | Scale Name | Minimum | Maximum | Data Value | Value_ratio |
|---------------|-------------------------|------------|---------|---------|------------|-------------|
| Statisticians | Writing                 | Importance | 1       | 5       | 3.50       | 0.700       |
| Statisticians | Reading Comprehension   | Importance | 1       | 5       | 4.00       | 0.800       |
| ...           | ...                     | ...        | ...     | ...     | ...        | ...         |

# 5. Technology Skills: Title-Example (No element and Scale)

[Contents data(Sample)]

| O*NET-SOC Code | Title         | Example                       | Commodity Code | Commodity Title                        | Hot Technology In Demand |
|----------------|---------------|-------------------------------|----------------|----------------------------------------|--------------------------|
| 15-2041.00     | Statisticians | Amazon Redshift               | 43232306       | Data base user interface and query...  | Y                        |
| ...            | ...           | ...                           | ...            | ...                                    | ...                      |

# 6. Tools Used: Title-Example (No element and Scale)

[Contents data(Sample)]

| O*NET-SOC Code | Title         | Example            | Commodity Code | Commodity Title    |
|----------------|---------------|--------------------|----------------|---------------------|
| 15-2041.00     | Statisticians | Desktop computers | 43211507       | Desktop computers  |
| ...            | ...           | ...                | ...            | ...                 |

# 7. Work Activities: Title-Element-Scale

[Contents data(Sample)]

| Title         | Element Name                      | Scale Name | Minimum | Maximum | Data Value | Value_ratio |
|---------------|-----------------------------------|------------|---------|---------|------------|-------------|
| Statisticians | Getting Information                | Importance | 1       | 5       | 4.5        | 0.9         |
| ...           | ...                               | ...        | ...     | ...     | ...        | ...         |

# 8. Work Context: Title-Element-Scale-Category(Nan, Category)

[Contents data(Sample)]

| Title         | Element Name                    | Scale Name          | Minimum | Maximum | Category | Category Description              | Data Value | Value_ratio |
|---------------|---------------------------------|---------------------|---------|---------|----------|----------------------------------|------------|-------------|
| Statisticians | Face-to-Face Discussions         | Context             | 1       | 5       | NaN      | NaN                              | 4.4        | 0.88        |
| ...           | ...                             | ...                 | ...     | ...     | ...      | ...                              | ...        | ...         |

# 9. Work Styles: Title-Element-Scale

[Contents data(Sample)]

| Title         | Element Name   | Scale Name | Minimum | Maximum | Data Value | Value_ratio |
|---------------|----------------|------------|---------|---------|------------|-------------|
| Statisticians | Persistence    | Importance | 1       | 5       | 4.0        | 0.8         |
| ...           | ...            | ...        | ...     | ...     | ...        | ...         |

# 10. Work Values: Title-Element-Scale

[Contents data(Sample)]

| Title         | Element Name   | Scale Name | Minimum | Maximum | Data Value | Value_ratio |
|---------------|----------------|------------|---------|---------|------------|-------------|
| Statisticians | Relationships  | Extent     | 1       | 7       | 4.67       | 0.667       |
| ...           | ...            | ...        | ...     | ...     | ...        | ...         |