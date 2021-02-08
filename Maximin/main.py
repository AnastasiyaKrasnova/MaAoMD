from Maximin import Maximin
from Plotting import Plotting
import time

def main():
    maximin=Maximin()
    maximin.init_data(20000,5)
    'Plotting.draw_plot(maximin.dataset, maximin.labels, True)'
    maximin.generate_center()
    maximin.find_second_center()
    while True:
        maximin.divide_to_centers()
        maximin.move_centers()
        res=maximin.new_kernel()
        if (res==False):
            maximin.divide_to_centers()
            maximin.move_centers()
            break

    Plotting.draw_plot(maximin.dataset, maximin.labels, False, maximin.centers)

if __name__ == "__main__":
	main()