import matplotlib.pyplot as plt
import decimal
import numpy as np

SIZE=1000
def generateFunctionData(params, min, max):
    xmin = min
    xmax = max
    dx = 0.01
    xlist = np.around(np.arange(xmin, xmax, dx), decimals=4)
    ylist = ((-1*params[1]*xlist) + (-1*params[0])) / (params[3]*xlist + params[2])
    return xlist, ylist

def generateTestData():
    xmin = -4.5
    xmax = 4.5
    ymin = -19
    ymax = 19
    xlist = np.random.uniform(low=xmin, high=xmax, size=(SIZE,))
    ylist = np.random.uniform(low=ymin, high=ymax, size=(SIZE,))
    return xlist, ylist


def viewData(params,x_test,y_test,labels,src_points):
    x,y=generateFunctionData(params,-5,5)

    threshold = 20
    y = np.ma.masked_less(y, -1 * threshold)
    y = np.ma.masked_greater(y, threshold)
    plt.plot(x,y,color="tab:orange")

    plt.scatter(x_test,y_test, c=labels, s=10)

    for i in range(len(src_points)):
        category, points= src_points[i]
        if (category==1):
            plt.scatter(points[0],points[1], c='blue', s=70, alpha=0.7)
        else:
             plt.scatter(points[0],points[1], c='red', s=70, alpha=0.7)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('zero'))
    ax.spines['left'].set_position(('zero'))
    plt.grid(True)
    plt.show()