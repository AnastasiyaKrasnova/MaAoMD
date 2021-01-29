import seaborn as sns; sns.set();
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from random import seed
from random import randint
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin


def find_my_clusters(dataset, n_clusters):
    rgn=np.random.RandomState(randint(0,2**32-1))
    i=rgn.permutation(dataset.shape[0])[:n_clusters]
    centers=dataset[i]

    while True:
        labels = pairwise_distances_argmin(dataset, centers)
        new_centers = np.array([dataset[labels == j].mean(0)
                                for j in range(n_clusters)])
                                
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

def find_inter_clusters(dataset, n_clusters):
    kmeans=KMeans(n_clusters)
    kmeans.fit(dataset)
    y_kmeans=kmeans.predict(dataset)
    cluster_centers=kmeans.cluster_centers_
    return cluster_centers, y_kmeans

time_seed=round(time.time() * 1000)
seed(time_seed)

dataset, cluster_num_true=make_blobs(n_samples=10000, centers=5,cluster_std=3.1, random_state=randint(0,100))
plt.scatter(dataset[:,0],dataset[:,1],s=1)
plt.show()

cluster_centers, y_kmeans=find_inter_clusters(dataset, 5)
plt.scatter(dataset[:,0],dataset[:,1], c=y_kmeans, s=1, cmap='viridis')
plt.scatter(cluster_centers[:,0],cluster_centers[:,1], c='black', s=50, alpha=0.5)
plt.show()

centers, labels=find_my_clusters(dataset, 5)
plt.scatter(dataset[:,0],dataset[:,1], c=labels, s=1, cmap='viridis')
plt.scatter(centers[:,0],centers[:,1], c='black', s=50, alpha=0.5)
plt.show()



