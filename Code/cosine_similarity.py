from sklearn.metrics.pairwise import cosine_similarity

# Convert job description embedding to numpy array
# Assuming job_description_embedding is a tensor
job_emb = job_description_embedding.numpy()

# Convert each resume embedding to numpy array
resume_embs = resume_embeddings.apply(lambda x: x.numpy())

# Calculating similarity for each resume
# Calculate similarities while retaining the original index
similarities = resume_embeddings.apply(lambda x: cosine_similarity(x.detach().numpy().squeeze().reshape(1, -1), job_emb.reshape(1, -1))[0][0])

# Setting the same index as the original DataFrame to align similarities with resume IDs
similarities.index = resumes_with_id.index

# Getting the indices of the top matching resumes
top_match_indices = similarities.nlargest(3).index

# Retrieving the IDs of the top matching resumes
top_match_ids = resumes_with_id.loc[top_match_indices, 'ID']

print(top_match_ids)
# %%
##########################################################################

# Calculate similarity (TFIDF)
for i, job_vec in enumerate(job_desc_vec):
    similarities = cosine_similarity(job_vec, resumes_vec)
    top_resume_indices = similarities.argsort()[0][-3:]  # Gets top 3 resumes
    top_resumes = df.iloc[top_resume_indices]['ID']

print(f"Top resumes for job {i}: {top_resumes.values}")

