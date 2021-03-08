from data import *
from Plotting import *
from Potentials import PotentialMethod

def main():
    classifier=PotentialMethod(feature_data)
    params=classifier.train()
    xlist,ylist=generateTestData()
    labels=[]
    print(classifier.guess(0.5,2))
    print(classifier.guess(0.5,-2))
    print(xlist)
    print(ylist)
    for i in range(len(xlist)):
        func_class=classifier.guess(xlist[i],ylist[i])
        if (func_class==1):
            labels.append('blue')
        else:
            labels.append('red')
    viewData(params,xlist,ylist,np.array(labels))

if __name__ == "__main__":
    main()