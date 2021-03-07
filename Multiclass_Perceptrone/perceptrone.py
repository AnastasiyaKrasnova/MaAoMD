import numpy as np
import random

BIAS = 1                         
ITERATIONS = 1000             


class MultiClassPerceptron():
    def __init__(self, classes, feature_list, feature_data, iterations=ITERATIONS):
        self.classes = classes
        self.feature_list = feature_list
        self.feature_data = feature_data
        self.iterations = iterations

        random.shuffle(self.feature_data)
        self.train_set = self.feature_data[:int(len(self.feature_data))]
        self.weight_vectors = {c: np.array([0 for _ in range(len(self.feature_list) + 1)]) for c in self.classes}


    def train(self):
        for _ in range(self.iterations):
            for category, feature_dict in self.train_set:
                feature_list = [feature_dict[k] for k in range(len(self.feature_list))]
                feature_list.append(BIAS)
                feature_vector = np.array(feature_list)

                arg_max, predicted_class = 0, self.classes[0]

                for c in self.classes:
                    current_activation = np.dot(feature_vector, self.weight_vectors[c])
                    if current_activation >= arg_max:
                        arg_max, predicted_class = current_activation, c

                if not (category == predicted_class):
                    self.weight_vectors[category] += feature_vector
                    self.weight_vectors[predicted_class] -= feature_vector

    def predict(self, feature_dict):
        feature_list = feature_dict
        feature_list.append(BIAS)
        feature_vector = np.array(feature_list)

        arg_max, predicted_class = 0, self.classes[0]

        for c in self.classes:
            current_activation = np.dot(feature_vector, self.weight_vectors[c])
            if current_activation >= arg_max:
                arg_max, predicted_class = current_activation, c

        return predicted_class
    
    def printFuncs(self):
        print("desicion functions:")
        for c in self.classes:
            func="d_"+c+"(x)="
            for i in range(0,len(self.weight_vectors[c])):
                #print(i)
                if (self.weight_vectors[c][i]>=0 and i>0):
                    func+="+"+str(self.weight_vectors[c][i])
                else:
                    func+=str(self.weight_vectors[c][i])
                if (i<len(self.feature_list)):
                    func+="*x"+str(i)
            print(func)
