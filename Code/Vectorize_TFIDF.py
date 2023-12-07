from sklearn.feature_extraction.text import TfidfVectorizer

#%%

# job discription 
# for accounting assistant
"""Reconcile invoices and identify discrepancies. Create and update expense reports.Process reimbursement forms.
Prepare bank deposits. Enter financial transactions into internal databases. Check spreadsheets for accuracy.
Maintain digital and physical financial records."""

# Combine all texts for vectorization 
# it is a must to combine then vectorize then sperate

combined_texts = df['Cleaned_Resume'].tolist() + [job_description]
#%%

# Vectorization
vectorizer = TfidfVectorizer()
vectorized_texts = vectorizer.fit_transform(combined_texts)
#%%
# Splitting the vectorized texts back into resumes and job descriptions
resumes_vec = vectorized_texts[:len(df)]
job_desc_vec = vectorized_texts[len(df):]

