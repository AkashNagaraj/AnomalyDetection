# https://dev.to/sajal2692/coding-k-means-clustering-using-python-and-numpy-fg1
import numpy as np

def initialize_random_centroids(K, X):
    m, n = np.shape(X) # X - input data rows*features
    centroids = np.empty((K,n))
    for i in range(K): # K - number of centroids
        centroids[i] = X[np.random.choice(range(m))]
    return centroids


def euclidean_distance(x1,x2):
    return np.linalg.norm(x1-x2) #np.sqrt(np.sum(np.power(x1-x2,2)))


def closest_centroid(x, centroids, K):
    distances = np.empty(K)
    for i in range(K):
        distances[i] = euclidean_distance(centroids[i],x)
    return np.argmin(distances)


def create_clusters(centroids, K, X):
    m, _ = np.shape(X)
    cluster_idx = np.empty(m)
    for i in range(m):
        cluster_idx[i] = closest_centroid(X[i], centroids, K)
    return cluster_idx


def compute_means(cluster_idx, K, X):
    _, n = np.shape(X)
    centroids = np.empty((K,n))
    for i in range(K):
        points = X[cluster_idx==i]
        centroids[i] = np.mean(points, axis=0)
    return centroids


def run_Kmeans(K, X, y, max_iterations=5):
    centroids = initialize_random_centroids(K, X)

    for _ in range(max_iterations):
        clusters = create_clusters(centroids, K, X)
        previous_centroids = centroids
        centroids = compute_means(clusters, K, X)
        diff = previous_centroids - centroids
    return np.count_nonzero(clusters==y)/len(clusters)



