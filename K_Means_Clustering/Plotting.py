import seaborn as sns; sns.set();
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Plotting:
    @staticmethod
    def draw_plot(_dataset,_labels,is_init,_centers=0):
        if is_init:
            plt.scatter(_dataset[:,0],_dataset[:,1], s=1)
        else:
            plt.scatter(_dataset[:,0],_dataset[:,1], c=_labels, s=1, cmap='viridis')
            plt.scatter(_centers[:,0],_centers[:,1], c='black', s=50, alpha=0.5)
        plt.show()