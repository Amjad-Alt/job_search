#%%
# Use one resume (input on Streamlit)

# Read PDF, Make a String (text) , Dafaframe

df_resume['Resume'].to_csv("./job_zone_model_resume_test.csv")
# Transformer format: DatasetDict
data = load_dataset("csv", data_files="./job_zone_model_resume_test.csv")
data = prep_zone_model_test_resume(data)  # 2481

#%%
# predict on the test data (does not have a lable)

#%%
