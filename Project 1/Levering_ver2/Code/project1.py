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
    dir_acc = {1:[], 2:[], 3:[], 4:[]}

    for combination in all_combinations_of_features:
        train_set = data_with_specified(train_x, combination)
        test_set = data_with_specified(test_x, combination)

        c = Nearest_neighbor_Classifier()
        c.fit(train_set, train_t)
        acc = c.accuracy(test_set, test_t)
        lst = [acc, combination]

        dir_acc[len(combination)].append(lst)

    best_comb = []
    for key in dir_acc.keys():
        dim_lst = dir_acc[key]
        max_acc = 0.0
        comb = None
        for i in range(len(dim_lst)):
            if dim_lst[i][0] >= max_acc:
                max_acc = dim_lst[i][0]
                comb = dim_lst[i][1]
        if comb != None:
            best_comb.append(comb)


    return best_comb

def data_set_each_dim(data_set, combination_lst):
    data_lst = []
    for comb in combination_lst:
        data_lst.append(data_with_specified(data_set,comb))
    return data_lst

def find_nest_comb_with_given_classifier(cl_type,train_lst, test_lst, train_t_lst, test_t_lst):

    #for each data set
    dataset_as_key = {}
    for j in range(len(train_lst)):
        train_x = train_lst[j]
        test_x = test_lst[j]
        train_t = train_t_lst[j]
        test_t = test_t_lst[j]
        #for each dimention
        dimention_as_key = {}
        for i in range(len(train_x)):
            if cl_type == "Minimum_failrate_Classifier":
                c1 = Minimum_failrate_Classifier()
            elif cl_type == "Nearest_neighbor_Classifier":
                c1 = Nearest_neighbor_Classifier()
            else:
                c1 = Minimum_squared_error_Classifier()

            c1.fit(train_x[i],train_t)
            accuracy = c1.accuracy(test_x[i], test_t)
            error = c1.error_rate(test_x[i],test_t)
            dimention_as_key[i+1] = [cl_type, accuracy, error]

        dataset_as_key[j+1] = dimention_as_key
    return dataset_as_key

train_x1, train_t1, test_x1, test_t1 = read_and_create_data_set("data/ds-1.txt")
train_x2, train_t2, test_x2, test_t2 = read_and_create_data_set("data/ds-2.txt")
train_x3, train_t3, test_x3, test_t3 = read_and_create_data_set("data/ds-3.txt")

best_combination_of_dataset_1 = find_best_combination(train_x1, train_t1, test_x1, test_t1)
best_combination_of_dataset_2 = find_best_combination(train_x2, train_t2, test_x2, test_t2)
best_combination_of_dataset_3 = find_best_combination(train_x3, train_t3, test_x3, test_t3)

print("--------------------------------------")
print("Best combination of data set 1: ", best_combination_of_dataset_1)
print("Best combination of data set 2: ", best_combination_of_dataset_2)
print("Best combination of data set 3: ", best_combination_of_dataset_3)
print("--------------------------------------")


#Remake data matrix after the best combination
train_x1_lst = data_set_each_dim(train_x1, best_combination_of_dataset_1)
test_x1_lst = data_set_each_dim(test_x1, best_combination_of_dataset_1)

train_x2_lst = data_set_each_dim(train_x2, best_combination_of_dataset_2)
test_x2_lst = data_set_each_dim(test_x2, best_combination_of_dataset_2)

train_x3_lst = data_set_each_dim(train_x3, best_combination_of_dataset_3)
test_x3_lst = data_set_each_dim(test_x3, best_combination_of_dataset_3)

train = [train_x1_lst,train_x2_lst,train_x3_lst]
test = [test_x1_lst,test_x2_lst,test_x3_lst]
train_t = [train_t1,train_t2,train_t3]
test_t = [test_t1, test_t2,test_t3]

mfc_dir = find_nest_comb_with_given_classifier("Minimum_failrate_Classifier",train, test, train_t, test_t)
nnc_dir = find_nest_comb_with_given_classifier("Nearest_neighbor_Classifier",train, test, train_t, test_t)
msec_dir = find_nest_comb_with_given_classifier("Minimum_squared_error_Classifier",train, test, train_t, test_t)

for data_set_number in mfc_dir.keys():
    #in data set
    mfc_data_set_n = mfc_dir[data_set_number]
    nnc_data_set_n = nnc_dir[data_set_number]
    msec_data_set_n = msec_dir[data_set_number]
    for dim in mfc_data_set_n.keys():
        mfc_data_dim_n = mfc_data_set_n[dim]
        nnc_data_dim_n = nnc_data_set_n[dim]
        msec_data_dim_n = msec_data_set_n[dim]
        if (mfc_data_dim_n[1] >= nnc_data_dim_n[1]) and (mfc_data_dim_n[1] >= msec_data_dim_n[1]):
            print("----------------------------------------------")
            print("Data set: ", data_set_number)
            print("Dimensjon: ", dim)
            print("Beste klassifikator er ", mfc_data_dim_n[0])
            print("Accuracy: ", mfc_data_dim_n[1])
            print("Error: ", mfc_data_dim_n[2])
            print()
        elif (nnc_data_dim_n[1] >= mfc_data_dim_n[1]) and (nnc_data_dim_n[1] >= msec_data_dim_n[1]):
            print("----------------------------------------------")
            print("Data set: ", data_set_number)
            print("Dimensjon: ", dim)
            print("Beste klassifikator er ", nnc_data_dim_n[0])
            print("Accuracy: ", nnc_data_dim_n[1])
            print("Error: ", nnc_data_dim_n[2])
            print()
        else:
            print("----------------------------------------------")
            print("Data set: ", data_set_number)
            print("Dimensjon: ", dim)
            print("Beste klassifikator er ", msec_data_dim_n[0])
            print("Accuracy: ", msec_data_dim_n[1])
            print("Error: ", msec_data_dim_n[2])
            print()
