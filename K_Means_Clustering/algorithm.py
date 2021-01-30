from MyKMeans import MyKMeans
from Plotting import Plotting

def main():
    kmeance=MyKMeans()
    kmeance.init_data(20000,5,5)
    Plotting.draw_plot(kmeance.dataset, kmeance.labels, True)
    kmeance.find_inter_clusters()
    Plotting.draw_plot(kmeance.dataset, kmeance.labels, False, kmeance.centers)
    kmeance.find_my_clusters()
    Plotting.draw_plot(kmeance.dataset, kmeance.labels, False, kmeance.centers)

if __name__ == "__main__":
	main()




