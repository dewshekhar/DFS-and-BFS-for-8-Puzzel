import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset from the provided GitHub link
url = "https://raw.githubusercontent.com/dewshekhar/DFS-and-BFS-for-8-Puzzel/main/cancer.csv"
data = pd.read_csv(url)

# Drop non-numeric columns and constant columns
data_numeric = data.select_dtypes(include=[np.number])
data_numeric = data_numeric.loc[:, ~data_numeric.columns.duplicated()]
data_numeric = data_numeric.dropna(axis=1, how='all')  # Drop columns with all missing values

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(data_numeric)
data_imputed = pd.DataFrame(data_imputed, columns=data_numeric.columns)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Number of clusters
num_clusters = 2

# K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)  # Explicitly set n_init
kmeans_labels = kmeans.fit_predict(data_scaled)

# Visualize the clusters and count the number of points in each cluster
plt.figure(figsize=(8, 6))

# Select "radius_mean" and "texture_mean" for plotting
X_plot = data_imputed[['radius_mean', 'texture_mean']].values

# Plot clusters
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5)

# Count the number of points in each cluster
cluster_counts = np.bincount(kmeans_labels)
for i, count in enumerate(cluster_counts):
    plt.text(X_plot[i, 0], X_plot[i, 1], str(count), color='black', fontweight='bold')

plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.title('K-Means Clustering with Cluster Counts')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.show()
