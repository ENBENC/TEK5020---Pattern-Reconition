import numpy as np
import math
from functions import *
from numba import jit

class Minimum_failrate_Classifier():
    def __init__(self):
        self.sigma_hat = None;
        self.mu_hat = None;
        self.x_train = None;
        self.t_train = None;
        self.mu = None;
        self.W = None;
        self.w = None;
        self.w_0 = None;

    def return1(self):
        return 1

    def P_omega(self, t_train, taget_class):
        counter = 0

        for class_lab in t_train:
            if class_lab == taget_class:
                counter += 1

        return counter/len(t_train)

    def predict(self, data_point):
        g_i = []
        class_tag = []
        data_point = data_point.reshape(-1,1)

        for key in self.sigma_hat:
            class_tag.append(key)

        for i in range(len(self.mu)):
            #Append the g_i value to list and wait for final calculation
            tmp = data_point.T.dot(self.W[i])
            g_i.append(tmp.dot(data_point) + self.w[i].T.dot(data_point)+self.w_0[i])

        max_value = max(g_i)
        max_index = g_i.index(max_value)

        #Rturn the predict class value
        return class_tag[max_index]

    def fit(self, x_train, t_train):
        """
        1. Stplit x_train in to c list after numbers of class, and save them in a numpy array
        2. Estimate mu_i and sigma_i, and have them in a form of vectors
        Finish
        """

        self.x_train = x_train
        self.t_train = t_train

        class_dir = same_class_train_set(x_train,t_train)

        #NB! data point x and mu is colunm vector

        #Estimation of mu_hat using the train set
        #mu_hat = {1:mu1, 2:mu2, 3:mu3, ...}
        self.mu_hat = {}
        for key in class_dir:
            mu_i = (1/len(class_dir[key]))*np.sum(class_dir[key], axis=0)
            self.mu_hat[key] = mu_i.reshape(-1,1)
        #self.mu_hat = (1/len(x_train))*np.sum(x_train,axis=0)

        #Estimation of sigma_hat using the train set and mu_hat
        #sigma_hat = {1:sigma1, 2:sigma2, 3:sigma3, ...}
        self.sigma_hat = {}
        for key in class_dir:
            sum = np.zeros(shape=(self.mu_hat[key].shape[0],self.mu_hat[key].shape[0]))
            for x in class_dir[key]:
                x = x.reshape(-1,1)
                sum += (x - self.mu_hat[key])*((x - self.mu_hat[key]).reshape(1,-1)[0])

            sigma_i = (1/len(class_dir[key]))*sum
            self.sigma_hat[key] = sigma_i

        mu,W,w,w_0 = [],[],[],[]
        for key in self.sigma_hat:
            sigma_i = self.sigma_hat[key]
            #MAYBE TRANSPOSE MU_I akse monday!!!!!!
            mu_i = self.mu_hat[key]
            #x*W*x = cosntant
            W_i = (-1/2)*np.linalg.inv(sigma_i)
            #w_i *x = constant
            w_i = np.linalg.inv(sigma_i).dot(mu_i)

            #cosntant
            tmp = (-1/2)*mu_i.reshape(1,-1).dot(np.linalg.inv(sigma_i))
            w_i0 = tmp.dot(mu_i)[0][0] \
                - (1/2)*math.log(abs(np.linalg.det(sigma_i))) + math.log(self.P_omega(self.t_train,key))

            mu.append(mu_i)
            W.append(W_i)
            w.append(w_i)
            w_0.append(w_i0)

        self.mu = np.array(mu)
        self.W = np.array(W)
        self.w = np.array(w)
        self.w_0 = np.array(w_0)

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
