import pandas as pd

# Load IRIS dataset
iris_df = pd.read_csv("D:\Iris.csv")

# Remove Class Label column
iris_data = iris_df.drop(columns=["Class"])

# Convert dataframe to numpy array
iris_data = iris_data.values

# Apply k-Means clustering
iris_clusters, iris_centroids = k_means(iris_data, k)

# Print clusters and centroids
print("IRIS Clusters:", iris_clusters)
print("IRIS Centroids:", iris_centroids)
