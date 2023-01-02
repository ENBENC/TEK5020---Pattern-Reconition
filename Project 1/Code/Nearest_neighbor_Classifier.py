import numpy as np
import math
class Nearest_neighbor_Classifier():
    def __init__(self):
        self.x_train = None
        self.t_train = None

    def predict(self, data_point):

        #find distance between data_point and all other points in train set
        dist_lst = []
        for x in self.x_train:
            tmp_x = data_point - x
            tmp_x = np.sum(tmp_x**2)
            dist = math.sqrt(tmp_x)
            dist_lst.append(dist)

        #Get the minimum distance from the distance list,
        #and then get index to the point.
        #The index to the point corresponding to the point in x_train,
        #and to the class label in t_train.
        min_value = min(dist_lst)
        min_index = dist_lst.index(min_value)
        return self.t_train[min_index]

    def fit(self, x_train, t_train):
        self.x_train = x_train
        self.t_train = t_train

    def error_rate(self, test_x, test_t):
        return round((1 - self.accuracy(test_x, test_t)),2)

    def accuracy(self, test_x, test_t):
        correct = 0
        pred_lst = []

        for i in range(len(test_t)):
            pred_value = self.predict(test_x[i])
            pred_lst.append(pred_value)

            if pred_value == test_t[i]:
                correct += 1

        return round((correct/len(test_x)),2)
