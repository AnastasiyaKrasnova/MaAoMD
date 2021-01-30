from Plotting import Plotting
import numpy as np
from sklearn.datasets import make_blobs
from random import seed
from random import randint
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

class MyKMeans:

    def init_data(self, _n_samples, _n_clusters, _n_centers):
        time_seed=round(time.time() * 1000)
        seed(time_seed)
        self.dataset, self.labels=make_blobs(n_samples=_n_samples, centers=_n_centers,cluster_std=1, random_state=randint(0,100))
        self.n_clusters=_n_clusters
        

    def find_my_clusters(self):
        rgn=np.random.RandomState(randint(0,2**32-1))
        i=rgn.permutation(self.dataset.shape[0])[:self.n_clusters]
        self.centers=self.dataset[i]
        init=False
        while True:
            self.labels = pairwise_distances_argmin(self.dataset, self.centers)
            new_centers = np.array([self.dataset[self.labels == j].mean(0)
                                for j in range(self.n_clusters)])
                                
            if np.all(self.centers == new_centers):
                break
            self.centers = new_centers
            if (init==False):
                Plotting.draw_plot(self.dataset, self.labels, False, self.centers,)
                init=True

    def find_inter_clusters(self):
        kmeans=KMeans(self.n_clusters)
        kmeans.fit(self.dataset)
        self.labels=kmeans.predict(self.dataset)
        self.centers=kmeans.cluster_centers_

