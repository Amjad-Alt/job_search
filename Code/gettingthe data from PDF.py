
# %%
import os
import PyPDF2
import pandas as pd
from pdfminer.high_level import extract_text
# %%


def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            # Handles pages where text extraction returns None
            text += page.extract_text() or ""
        return text


def process_pdfs(root_directory):
    data = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                category = os.path.basename(root)
                pdf_text = extract_text_from_pdf(file_path)
                pdf_id = os.path.splitext(file)[0]
                data.append(
                    {'ID': pdf_id, 'Resume': pdf_text, 'Category': category})
    return pd.DataFrame(data)


# %%
# root directory + prosessing the data to df
ACCOUNTANT = "data\ACCOUNTANT"
ACCOUNTANT_df = process_pdfs(ACCOUNTANT)
ADVOCATE = "data\ADVOCATE"
ADVOCATE_df = process_pdfs(ADVOCATE)
AGRICULTURE = "data\AGRICULTURE"
AGRICULTURE_df = process_pdfs(AGRICULTURE)
APPAREL = "data\APPAREL"
APPAREL_df = process_pdfs(APPAREL)
ARTS = "data\ARTS"
ARTS_df = process_pdfs(ARTS)
AUTOMOBILE = "data\AUTOMOBILE"
AUTOMOBILE_df = process_pdfs(AUTOMOBILE)
AVIATION = "data\AVIATION"
AVIATION_df = process_pdfs(AVIATION)
BANKING = "data\BANKING"
BANKING_df = process_pdfs(BANKING)
BPO = "data\BPO"
BPO_df = process_pdfs(BPO)
BUSINESS_DEVELOPMENT = "data\BUSINESS-DEVELOPMENT"
BUSINESS_DEVELOPMENT_df = process_pdfs(BUSINESS_DEVELOPMENT)
CHEF = "data\CHEF"
CHEF_df = process_pdfs(CHEF)
CONSTRUCTION = "data\CONSTRUCTION"
CONSTRUCTION_df = process_pdfs(CONSTRUCTION)
CONSULTANT = "data\CONSULTANT"
CONSULTANT_df = process_pdfs(CONSULTANT)
DESIGNER = "data\DESIGNER"
DESIGNER_df = process_pdfs(DESIGNER)
DIGITAL_MEDIA = "data\DIGITAL-MEDIA"
DIGITAL_MEDIA_df = process_pdfs(DIGITAL_MEDIA)
ENGINEERING = "data\ENGINEERING"
ENGINEERING_df = process_pdfs(ENGINEERING)
FINANCE = "data\FINANCE"
FINANCE_df = process_pdfs(FINANCE)
FITNESS = "data\FITNESS"
FITNESS_df = process_pdfs(FITNESS)
HEALTHCARE = "data\HEALTHCARE"
HEALTHCARE_df = process_pdfs(HEALTHCARE)
HR = "data\HR"
HR_df = process_pdfs(HR)
INFORMATION_TECHNOLOGY = "data\INFORMATION-TECHNOLOGY"
INFORMATION_TECHNOLOGY_df = process_pdfs(INFORMATION_TECHNOLOGY)
PUBLIC_RELATIONS = "data\PUBLIC-RELATIONS"
PUBLIC_RELATIONS_df = process_pdfs(PUBLIC_RELATIONS)
SALES = "data\SALES"
SALES_df = process_pdfs(SALES)
TEACHER = "data\TEACHER"
TEACHER_df = process_pdfs(TEACHER)


# check
print(ACCOUNTANT_df)
print(ARTS_df)
print(SALES_df)
print(TEACHER_df)
# %%
# List of all DataFrames
dfs = [
    ACCOUNTANT_df, ADVOCATE_df, AGRICULTURE_df, APPAREL_df, ARTS_df,
    AUTOMOBILE_df, AVIATION_df, BANKING_df, BPO_df, BUSINESS_DEVELOPMENT_df,
    CHEF_df, CONSTRUCTION_df, CONSULTANT_df, DESIGNER_df, DIGITAL_MEDIA_df,
    ENGINEERING_df, FINANCE_df, FITNESS_df, HEALTHCARE_df, HR_df,
    INFORMATION_TECHNOLOGY_df, PUBLIC_RELATIONS_df, SALES_df, TEACHER_df
]

# Concatenating all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)
# %%
# save the DataFrame to a CSV file
combined_df.to_csv('resumes_data.csv', index=False)
