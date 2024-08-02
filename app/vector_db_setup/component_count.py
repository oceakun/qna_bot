import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('simple_data.csv', delimiter=',', quotechar='"')

df['text'] = df['question']

# Create TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])

# Get the number of features (vocabulary size)
n_features = tfidf_matrix.shape[1]
print(f"Number of features in TF-IDF matrix: {n_features}")

# Adjust the range based on the number of features
max_components = min(n_features - 1, 1000)  # -1 to avoid potential issues
n_components_range = range(10, max_components + 1, 10)

explained_variances = []

for n_components in n_components_range:
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(tfidf_matrix)
    explained_variances.append(sum(svd.explained_variance_ratio_))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, explained_variances)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance vs. Number of LSA Components')
plt.grid(True)
plt.show()

# Find the number of components that explain 95% of the variance
threshold = 0.95
optimal_components = next((i for i, v in enumerate(explained_variances) if v >= threshold), len(explained_variances) - 1)
optimal_n_components = n_components_range[optimal_components]
print(f"Number of components explaining {threshold*100}% of variance: {optimal_n_components}")

# If 95% threshold wasn't reached, suggest using all components
if optimal_components == len(explained_variances) - 1 and explained_variances[-1] < threshold:
    print(f"Note: The {threshold*100}% threshold wasn't reached. Consider using all {max_components} components.")
    optimal_n_components = max_components

print(f"Suggested number of components to use: {optimal_n_components}")