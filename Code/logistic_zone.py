
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.metrics import plot_confusion_matrix




# Load and preprocess data
def create_zone_model_data():
    url = 'https://raw.githubusercontent.com/Amjad-Alt/job_search/Nammin-Woo/Data_cleaned/df_Occupation.csv'
    df_job = pd.read_csv(url)
    df_job.dropna(subset=['Job Zone'], inplace=True)

    # Concatenate the text columns
    text_columns = ['Description', 'Description_Abilities',
                    'Description_Knowledge', 'Description_Skills']
    df_job['combined_text'] = df_job[text_columns].fillna(
        '').agg(' '.join, axis=1)

    df_job['Job Zone'] = df_job['Job Zone'].astype('int64') - 1
    df_job.rename(columns={'Job Zone': 'label'}, inplace=True)
    df_job = df_job[['combined_text', 'label']]
    return df_job


# Preprocess data
df_job = create_zone_model_data()

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_job['combined_text'], df_job['label'], test_size=0.3, random_state=42)

# Feature extraction with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Logistic Regression Model
log_reg = LogisticRegression()

# Define grid search parameters
param_grid = {'C': [0.1, 1, 10], 'max_iter': [100, 200]}

# Grid search
grid_search = GridSearchCV(log_reg, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, train_labels)

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate on test set
best_model = grid_search.best_estimator_


# Define range of sample sizes to plot learning curve
train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# Calculate learning curve scores for both training and test 
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, train_labels, train_sizes=train_sizes, cv=5
)

# Calculate mean and standard deviation of training scores and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()


# Generate the confusion matrix plot
plot_confusion_matrix(best_model, X_test, test_labels, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

