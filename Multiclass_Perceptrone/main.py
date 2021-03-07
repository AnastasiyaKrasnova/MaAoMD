from data import *
from perceptrone import MultiClassPerceptron

def main():
    shape_classifier = MultiClassPerceptron(shape_classes, shape_feature_list, shape_feature_data)
    shape_classifier.train()
    print(shape_classifier.predict([250, 120, 20]))
    shape_classifier.printFuncs()

if __name__ == "__main__":
    main()
