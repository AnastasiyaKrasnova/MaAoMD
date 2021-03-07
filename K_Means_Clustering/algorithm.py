from MyKMeans import MyKMeans
from Plotting import Plotting
import time

def main():
    kmeance=MyKMeans()
    kmeance.init_data(10000,5,5)
    Plotting.draw_plot(kmeance.dataset, kmeance.labels, True)
    kmeance.find_inter_clusters()
    Plotting.draw_plot(kmeance.dataset, kmeance.labels, False, kmeance.centers)
    kmeance.find_my_clusters()
    Plotting.draw_plot(kmeance.dataset, kmeance.labels, False, kmeance.centers)

if __name__ == "__main__":
	main()

if(max_dist>self.find_mean_centers_dist()):
            self.centers=np.append(self.centers,nklist[np.argmax(np.array(nkdist))].reshape(1,-1),0)
            self.n_clusters+=1



