from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

#%%
# Calculate similarity
for i, job_vec in enumerate(job_desc_vec):
    similarities = cosine_similarity(job_vec, resumes_vec)
    top_resume_indices = similarities.argsort()[0][-3:]  # Gets top 3 resumes
    top_resumes = df.iloc[top_resume_indices]['ID']
    print(f"Top resumes for job {i}: {top_resumes.values}")
