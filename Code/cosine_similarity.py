
from sklearn.metrics.pairwise import cosine_similarity
# Calculate similarities (bert)

# Assuming you want to compare with the first job description
job_emb = job_description_embeddings.iloc[0].numpy()
resume_embs = resume_embeddings.apply(lambda x: x.numpy())

# Calculating similarity for each resume
similarities = resume_embs.apply(
    lambda x: cosine_similarity([x], [job_emb])[0][0])

# Getting the top matching resumes
top_matches = similarities.nlargest(3)  # Adjust the number as needed

# %%
# Calculate similarity (TFIDF)
for i, job_vec in enumerate(job_desc_vec):
    similarities = cosine_similarity(job_vec, resumes_vec)
    top_resume_indices = similarities.argsort()[0][-3:]  # Gets top 3 resumes
    top_resumes = df.iloc[top_resume_indices]['ID']
    print(f"Top resumes for job {i}: {top_resumes.values}")
