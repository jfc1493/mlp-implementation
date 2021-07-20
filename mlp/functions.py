'''
Created on Nov 27, 2017

@author: Jean-Francis Carignan
'''

import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    x = np.atleast_2d(np.array(x))
    max_vals = np.amax(x, axis=1)
    max_vals = max_vals.reshape(max_vals.shape[0], 1)
    e = np.exp(x - max_vals)
    result = e / np.sum(e, axis=1, keepdims=True)
    return result


def reLU(x):
    return np.maximum(x, np.zeros(x.shape))


def randomWeights(l1, l2):
    weights = np.random.rand(l2, l1)
    weights = (weights-0.5)*2/l1
    return weights


def onehot(y, m):
    y = np.atleast_2d(y).transpose()
    onehot = np.zeros((len(y), m))
    for i in range(len(y)):
        onehot[i, y[i].astype(int)] = 1
    return onehot


def selectmax(y):
    indices = np.argmax(y, axis=1)
    return onehot(indices, y.shape[1])


def plot_decision(X, y, nn, res=1):

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res),
                           np.arange(x2_min, x2_max, res))

    Z = np.argmax(nn.compute_predictions(
        np.c_[xx1.ravel(), xx2.ravel()]), axis=1)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y,  alpha=1)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.show()


def plot_stats(train_err, valid_err, test_err, train_cost, valid_cost, test_cost):

    path = '.'

    # Errors
    plt.figure(1)
    plt.plot(train_err, '-g', label='train', marker='o')
    plt.plot(valid_err, '-r', label='valid', marker='o')
    plt.plot(test_err, '-b', label='test', marker='o')
    plt.title("Errors at each epoch")
    plt.legend()
    plt.savefig(path+'errors.png')

    # Costs
    plt.figure(2)
    plt.plot(train_cost, '-g', label='train', marker='o')
    plt.plot(valid_cost, '-r', label='valid', marker='o')
    plt.plot(test_cost, '-b', label='test', marker='o')
    plt.title("Costs at each epoch")
    plt.legend()
    plt.savefig(path+'costs.png')

    text = np.transpose((train_err, train_cost, valid_err,
                         valid_cost, test_err, test_cost))
    header = "train_err,train_cost,valid_err,valid_cost,test_err,test_cost"
    np.savetxt(path+'Errors and costs', text,
               fmt='%1.3f', delimiter=",", header=header)
