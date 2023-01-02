import numpy as np
def same_class_train_set(x_train,t_train):
    """
    Ex:
        The return dictionary looks like
        {1: [[feratur1 feature2 feature3] [feratur1 feature2 feature3] [feratur1 feature2 feature3]]
         2 : [[feratur1 feature2 feature3] [feratur1 feature2 feature3] [feratur1 feature2 feature3]]}
    Args:
        x_train(numpy array) : NxM numpy matrix
        t_train(numpy array) : 1xM numpy vector of target to x_train

    Returns:
        dir(python dictionary) : a dictionary with target(class lable)
                                as key, and alle point to the taget as a numpy array
    """
    dir = {}
    for i in range(len(t_train)):
        if t_train[i] in dir:
            dir[t_train[i]].append(x_train[i])
        else:
            dir[t_train[i]] = [x_train[i]]

    #Change alle python list to numpy array
    for key in dir:
        dir[key] = np.array(dir[key])

    return dir

def Y(X):
    Y_matrix = []
    for x in X:
        Y_matrix.append(np.append(1, x))
    return np.array(Y_matrix)
