from sklearn.feature_extraction.text import TfidfVectorizer

#%%

# job discription 

# Combine all texts for vectorization 
# it is a must to combine then vectorize then sperate

combined_texts = df['Cleaned_Resume'].tolist() + job_descriptions['Description'].tolist()
#%%

# Vectorization
vectorizer = TfidfVectorizer()
vectorized_texts = vectorizer.fit_transform(combined_texts)
#%%
# Splitting the vectorized texts back into resumes and job descriptions
resumes_vec = vectorized_texts[:len(df)]
job_desc_vec = vectorized_texts[len(df):]

