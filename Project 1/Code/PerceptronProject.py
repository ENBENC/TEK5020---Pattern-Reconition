from functions import *
import numpy as np
from Perceptron import Perceptron
import matplotlib.pyplot as plt

def plot_decision_regions(X, t, clf=[], size=(8,6)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02 # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    plt.figure(figsize=size) # You may adjust this
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap = 'Paired')
    plt.scatter(X[:,0], X[:,1], c=t, s=20.0, cmap='Paired')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()

train_x1, train_t1, test_x1, test_t1 = read_and_create_data_set("data/ds-1.txt")

train_x1 = data_with_specified(train_x1, [0,1])
test_x1 = data_with_specified(test_x1, [0,1])
cl = Perceptron()
cl.fit(train_x1, train_t1, 1, 100000)

print("Accuracy:", cl.accuracy(test_x1,test_t1))



plot_decision_regions(test_x1,test_t1,cl)
