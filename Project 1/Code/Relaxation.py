import numpy as np
import math
class Relaxation():
    def __init__(self):
        self.x_train = None
        self.t_train = None
        self.a = None
        self.b = None

    def predict(self, data_points, pos_class):
        pred = []
        for data_point in data_points:
            if self.a.dot(np.array([1,data_point[0],data_point[1]]).T) > self.b:
                pred.append(pos_class)
            else:
                pred.append(0)
        return np.array([pred])[0]

    def fit(self, x_train, t_train, pos_class, b, p, iterations):
        self.x_train = x_train
        self.t_train = t_train
        data_matrix = Y_target(self.x_train, self.t_train, pos_class)

        #b = np.ones(self.x_train[0].shape)
        a = np.zeros(data_matrix[0].shape)
        wrong_lst = []
        i = 0

        while i < iterations:
            for j in range(len(data_matrix)):
                if a.dot(data_matrix[j].T) <= b:
                    wrong_lst.append(data_matrix[j])
            sum = np.zeros(data_matrix[i].shape)
            for j in range(len(worng_lst)):
                sum += ((b-a.dot(worng_lst[j].T))/np.linalg.norm(worng_lst[j])^2)*worng_lst[j]

            a = a + p*sum
            i += 1
        self.a = a

    def error_rate(self, test_x, test_t):
        return round((1 - self.accuracy(test_x, test_t)),2)

    def accuracy(self, test_x, test_t):
        correct = 0
        pred_lst = self.predict(test_x)

        for i in range(len(test_t)):
            if pred_lst[i] == test_t[i]:
                correct += 1

        return round((correct/len(test_x)),2)
