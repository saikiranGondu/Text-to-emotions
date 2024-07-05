import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def k_means(data, k, max_iterations=100):
    # Randomly initialize centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    num_objects = data.shape[0]
    num_features = data.shape[1]
    
    # Initialize clusters and distances
    clusters = np.zeros(num_objects)
    distances = np.zeros((num_objects, k))
    
    # Main loop
    for _ in range(max_iterations):
        # Assign objects to the closest centroid
        for i in range(k):
            distances[:, i] = np.apply_along_axis(lambda x: euclidean_distance(x, centroids[i]), 1, data)
        clusters = np.argmin(distances, axis=1)
        
        # Update centroids
        for i in range(k):
            centroids[i] = np.mean(data[clusters == i], axis=0)
    
    return clusters, centroids

# Data objects
data_objects = np.array([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]])

# Applying k-Means clustering
k = 3
clusters, centroids = k_means(data_objects, k)

# Print clusters and centroids
print("Clusters:", clusters)
print("Centroids:", centroids)
