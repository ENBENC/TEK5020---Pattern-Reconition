import numpy as np
from Minimum_failrate_Classifier import Minimum_failrate_Classifier
from Nearest_neighbor_Classifier import Nearest_neighbor_Classifier
from Minimum_squared_error_Classifier import Minimum_squared_error_Classifier
import matplotlib.pyplot as plt

def read_and_create_data_set(path):
    """
    Args:
        path(str) : path to file

    Returns:
        train_x(numpy array) : array with features
        train_t(numpy array) : array with class to each datapoint in train_x
        test_x(numpy array) : array with features
        test_t(numpy array) : array with class to each datapoint in test_x
    """
    def read_data(path):
        """
        Open a txt file and convert the element inside to a NxN numpy matrix

        Ex: [[class feature1 feature2 feature3]
            [class feature1 feature2 feature3]
            [class feature1 feature2 feature3]

        Args:
            path(str) : path to file

        Return:
            data_matrix(numpy array) : NxN matrix
        """
        lst = []
        with open(path) as file:
            for line in file:
                data_point = line.split()
                lst.append(data_point)

            #Convert alle elements into float-
            data_matrix = np.array(lst).astype(np.float64)

        return data_matrix

    def get_train_test_set(data_matrix):
        """
            Split the data_matrix into train-set and test-set
            Where odd element 1,3,5... is train-set
            Where even element 0,2,4... is test-set

            NB! Python start from index 0, therefor element 1 have index 0

            Args:
                data_matrix(numpy array) : NxN matrix

            Returns:
                train_x(numpy array) : array with features
                train_t(numpy array) : array with class to each datapoint in train_x
                test_x(numpy array) : array with features
                test_t(numpy array) : array with class to each datapoint in test_x
        """

        train_x = []
        train_t = []

        test_x = []
        test_t = []

        for i in range(len(data_matrix)):
            #Odd element have even index
            if(i % 2 == 0):
                train_x.append(data_matrix[i][1:])
                train_t.append(data_matrix[i][0])
            else:
                test_x.append(data_matrix[i][1:])
                test_t.append(data_matrix[i][0])

        return np.array(train_x), np.array(train_t).astype(int), np.array(test_x), np.array(test_t).astype(int)

    data_matrix = read_data(path)
    return get_train_test_set(data_matrix)

def combinations(lst):
    """
    Find all combinations of the elements in lst

    Args:
        lst(Python list) : list of elements
    Returns:
        combinations_lst(Python list) : list of list of all combinations
    """
    def comb(lst):
        if len(lst) == 0:
            return [[]]
        cs = []
        for c in comb(lst[1:]):
            cs += [c, c+[lst[0]]]
        return cs

    combinations_lst = comb(lst)[1:]

    for i in range(len(combinations_lst)):
        combinations_lst[i].reverse()
    return combinations_lst

def data_with_specified(data_matrix, feature_lst):
    lst = []
    for feature in feature_lst:
        lst.append(data_matrix[:,[feature]].reshape(1,-1)[0])
    lst = np.array(lst)
    return lst.T

def find_best_combination(train_x, train_t, test_x, test_t):
    features = list(range(train_x.shape[1]))
    all_combinations_of_features = combinations(features)
    dir_acc = {}

    for combination in all_combinations_of_features:
        train_set = data_with_specified(train_x, combination)
        test_set = data_with_specified(test_x, combination)

        c = Nearest_neighbor_Classifier()
        c.fit(train_set, train_t)
        acc = c.accuracy(test_set, test_t)
        dir_acc[acc] = combination

    #find the maximum accuracy
    max_acc = max(dir_acc.keys())
    combination = dir_acc[max_acc]
    return combination

train_x1, train_t1, test_x1, test_t1 = read_and_create_data_set("data/ds-1.txt")
train_x2, train_t2, test_x2, test_t2 = read_and_create_data_set("data/ds-2.txt")
train_x3, train_t3, test_x3, test_t3 = read_and_create_data_set("data/ds-3.txt")

best_combination_of_dataset_1 = find_best_combination(train_x1, train_t1, test_x1, test_t1)
best_combination_of_dataset_2 = find_best_combination(train_x2, train_t2, test_x2, test_t2)
best_combination_of_dataset_3 = find_best_combination(train_x3, train_t3, test_x3, test_t3)

#Remake data matrix after the best combination
train_x1 = data_with_specified(train_x1, best_combination_of_dataset_1)
test_x1 = data_with_specified(test_x1, best_combination_of_dataset_1)

train_x2 = data_with_specified(train_x2, best_combination_of_dataset_2)
test_x2 = data_with_specified(test_x2, best_combination_of_dataset_2)

train_x3 = data_with_specified(train_x3, best_combination_of_dataset_3)
test_x3 = data_with_specified(test_x3, best_combination_of_dataset_3)

print("--------------------------------------")
print("Best combination of data set 1: ", best_combination_of_dataset_1)
print("Best combination of data set 2: ", best_combination_of_dataset_2)
print("Best combination of data set 3: ", best_combination_of_dataset_3)
print("--------------------------------------")


print("--------------------------------------")
print("Minimum_failrate_Classifier")
c1 = Minimum_failrate_Classifier()
c1.fit(train_x1,train_t1)
print("Data 1: ")
print("Accuracy: ", c1.accuracy(test_x1, test_t1))
print("Error rate: ", c1.error_rate(test_x1,test_t1))
print()

c2 = Minimum_failrate_Classifier()
c2.fit(train_x2,train_t2)
print("Data 2: ")
print("Accuracy: ", c2.accuracy(test_x2, test_t2))
print("Error rate: ", c2.error_rate(test_x2,test_t2))
print()

c3 = Minimum_failrate_Classifier()
c3.fit(train_x3,train_t3)
print("Data 3: ")
print("Accuracy: ", c3.accuracy(test_x3, test_t3))
print("Error rate: ", c3.error_rate(test_x3,test_t3))
print()
print("--------------------------------------")

print("--------------------------------------")
print("Nearest_neighbor_Classifier")
c1 = Nearest_neighbor_Classifier()
c1.fit(train_x1,train_t1)
print("Data 1: ")
print("Accuracy: ", c1.accuracy(test_x1, test_t1))
print("Error rate: ", c1.error_rate(test_x1,test_t1))
print()

c2 = Nearest_neighbor_Classifier()
c2.fit(train_x2,train_t2)
print("Data 2: ")
print("Accuracy: ", c2.accuracy(test_x2, test_t2))
print("Error rate: ", c2.error_rate(test_x2,test_t2))
print()

c3 = Nearest_neighbor_Classifier()
c3.fit(train_x3,train_t3)
print("Data 3: ")
print("Accuracy: ", c3.accuracy(test_x3, test_t3))
print("Error rate: ", c3.error_rate(test_x3,test_t3))
print()
print("--------------------------------------")

print("--------------------------------------")
print("Minimum_squared_error_Classifier")
c1 = Minimum_squared_error_Classifier()
c1.fit(train_x1,train_t1)
print("Data 1: ")
print("Accuracy: ", c1.accuracy(test_x1, test_t1))
print("Error rate: ", c1.error_rate(test_x1,test_t1))
print()

c2 = Minimum_squared_error_Classifier()
c2.fit(train_x2,train_t2)
print("Data 2: ")
print("Accuracy: ", c2.accuracy(test_x2, test_t2))
print("Error rate: ", c2.error_rate(test_x2,test_t2))
print()

c3 = Minimum_squared_error_Classifier()
c3.fit(train_x3,train_t3)
print("Data 3: ")
print("Accuracy: ", c3.accuracy(test_x3, test_t3))
print("Error rate: ", c3.error_rate(test_x3,test_t3))
print()
print("--------------------------------------")

"""
#Plots
data = data_with_specified(train_x1, [0,1])
x = data[:,0]
y = data[:,1]
plt.figure(0)
plt.scatter(x,y,c=train_t1)

data = data_with_specified(train_x2, [1,2])
x = data[:,0]
y = data[:,1]
plt.figure(1)
plt.scatter(x,y,c=train_t2)

data = data_with_specified(train_x3, [0,1])
x = data[:,0]
y = data[:,1]
plt.figure(2)
plt.scatter(x,y,c=train_t3)
plt.show()
"""
