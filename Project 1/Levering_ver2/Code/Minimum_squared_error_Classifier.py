import numpy as np
import math
from functions import *
from numpy.linalg import inv

class Minimum_squared_error_Classifier():
    def __init__(self):
        self.x_train = None
        self.t_train = None
        self.a = None

    def predict(self, data_point):
        a = self.a
        y = np.append(1, data_point)
        g = a.T.dot(y)

        if g >= 0:
            return 1
        else:
            return 2

    def fit(self, x_train, t_train):
        self.x_train = x_train
        self.t_train = t_train

        #Make Y_matrix, that transform x_train from x-space into y-space
        Y_matrix = Y(x_train)

        #Make b vector
        b = []
        for label in t_train:
            if label == 1:
                b.append(1)
            else:
                b.append(-1)
        b = np.array(b)

        #Compute a vector, solution vector, normal on the decision boundary
        a = ((inv(Y_matrix.T.dot(Y_matrix))).dot(Y_matrix.T)).dot(b)
        self.a = a

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
