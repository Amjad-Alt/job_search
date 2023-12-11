from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

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

# MLP Model
mlp = MLPClassifier()

# Define grid search parameters
param_grid = {
    'hidden_layer_sizes': [(50,), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

# Grid search
grid_search = GridSearchCV(mlp, param_grid, cv=3,
                           scoring='accuracy', verbose=2)
grid_search.fit(X_train, train_labels)

# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate on test set
best_mlp_model = grid_search.best_estimator_

# Predict the test set results
y_pred = best_mlp_model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(test_labels, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Define range of sample sizes to plot learning curve
train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# Calculate learning curve scores for both training and test 
train_sizes, train_scores, val_scores = learning_curve(
    best_mlp_model, X_train, train_labels, train_sizes=train_sizes, cv=5
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
plot_confusion_matrix(best_mlp_model, X_test, test_labels, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
