import numpy as np
from sklearn.datasets import make_blobs
from random import seed
from random import randint
import time
from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial import distance

class Maximin:

    def init_data(self, _n_samples, _n_centers):
        time_seed=round(time.time() * 1000)
        seed(time_seed)
        'self.dataset, self.labels=make_blobs(n_samples=_n_samples, centers=_n_centers,cluster_std=1, random_state=randint(0,100))'
        self.dataset=np.random.rand(_n_samples,2)
        self.n_clusters=0

    def find_mean_centers_dist(self):
        dist=distance.cdist(self.centers, self.centers, 'euclidean')
        return dist.mean()/2 

    def find_second_center(self):
        dist=distance.cdist(self.centers.reshape(1,-1), self.dataset, 'euclidean')
        self.centers=np.append(self.centers.reshape(1,-1),self.dataset[np.argmax(dist)].reshape(1,-1),0)
        self.n_clusters+=1
        

    def generate_center(self):
        rgn=np.random.RandomState(randint(0,2**32-1))
        i=rgn.permutation(self.dataset.shape[0])[1]
        self.centers=self.dataset[i]
        self.n_clusters+=1
    
    def divide_to_centers(self):
        self.labels = pairwise_distances_argmin(self.dataset, self.centers)

    def move_centers(self):
        new_centers = np.array([self.dataset[self.labels == j].mean(0)
                                for j in range(self.n_clusters)])
        self.centers = new_centers

    def new_kernel(self):
        nklist = []
        nkdist=[]
        for j in range(self.n_clusters):
            ds=self.dataset[self.labels == j]
            dist=distance.cdist(self.centers[j].reshape(1,-1), ds, 'euclidean')
            nklist.append(ds[np.argmax(dist)].reshape(1,-1))
            nkdist.append(np.max(dist))
        max_dist=np.max(np.array(nkdist))
        if(max_dist >self.find_mean_centers_dist()):
            self.centers=np.append(self.centers,nklist[np.argmax(np.array(nkdist))].reshape(1,-1),0)
            self.n_clusters+=1
            return True
        else:
            return False
        

           

