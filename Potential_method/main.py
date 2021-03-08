from data import *
from Plotting import *
from Potentials import PotentialMethod

def main():
    classifier=PotentialMethod(feature_data)
    params=classifier.train()
    x_testlist,y_testlist=generateTestData()
    test_labels=[]
    for i in range(len(x_testlist)):
        func_class=classifier.guess(x_testlist[i],y_testlist[i])
        if (func_class==1):
            test_labels.append('blue')
        else:
            test_labels.append('red')
    viewData(params,x_testlist,y_testlist,np.array(test_labels), feature_data)

if __name__ == "__main__":
    main()