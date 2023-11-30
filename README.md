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

      - 10 DataFrames merged into Clean_data with final preprocessing   