'''
Created on Nov 28, 2017

@author: Jean-Francis Carignan
'''
from functions import randomWeights, reLU, softmax, onehot, selectmax, plot_stats

import numpy as np
import copy


class NN:
    # Initialisation du modele avec le nombre de neurones a chaque couche
    def __init__(self, n_input, n_hidden, n_output):

        # Parametres initialiser aleatoirement
        w1 = randomWeights(n_input, n_hidden)
        b1 = np.atleast_2d(np.zeros(n_hidden))
        w2 = randomWeights(n_hidden, n_output)
        b2 = np.atleast_2d(np.zeros(n_output))
        self.W = dict([('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2)])

        self.n_output = n_output

    def fit(self, x, y, K=1, epochs=25, lrate=0.03, wd=0.0001, fast=False, print_stats=False):

        train_err, valid_err, test_err, train_cost, valid_cost, test_cost = [], [], [], [], [], []
        for ep in range(epochs):
            for i in range(len(y)/K):
                if (fast == False):
                    grads = self.bprop_slow(x[i*K:i*K+K, :], y[i*K:i*K+K])
                else:
                    grads = self.bprop_fast(x[i*K:i*K+K, :], y[i*K:i*K+K])
                self.W['w1'] -= lrate * (grads['w1'] + wd * 2*self.W['w1'])
                self.W['b1'] -= lrate * grads['b1']
                self.W['w2'] -= lrate * (grads['w2'] + wd * 2*self.W['w2'])
                self.W['b2'] -= lrate * grads['b2']
            if (print_stats):
                print("-----Epoch " + str(ep) + "-----")
                print("--Errors--")
                train = self.compute_error(x, y)
                valid = self.compute_error(self.valid_data, self.valid_labels)
                test = self.compute_error(self.test_data, self.test_labels)
                train_err.append(train)
                valid_err.append(valid)
                test_err.append(test)
                print("Train: " + str(train))
                print("Valid: " + str(valid))
                print("Test: " + str(test))
                print("--Mean costs--")
                train = self.compute_loss(x, y)
                valid = self.compute_loss(self.valid_data, self.valid_labels)
                test = self.compute_loss(self.test_data, self.test_labels)
                train_cost.append(train)
                valid_cost.append(valid)
                test_cost.append(test)
                print("Train: " + str(train))
                print("Valid: " + str(valid))
                print("Test: " + str(test))

        if(print_stats):
            plot_stats(train_err, valid_err, test_err,
                       train_cost, valid_cost, test_cost)

    def compute_error(self, x, y):

        preds = selectmax(self.compute_predictions(x))
        labels = onehot(y, self.n_output)
        results = preds*labels

        return 1-results.sum()/float(len(y))

    def compute_predictions(self, x):
        return self.fprop(x)['o_s']

    def compute_loss(self, x, y):
        loss = np.sum(-np.log(self.compute_predictions(x))
                      * onehot(y, self.n_output))
        return loss/y.shape[0]

    def FD_gradients(self, x, y):

        eps = 1e-6

        L = self.compute_loss(x, y)

        gradients = copy.deepcopy(self.W)

        for param in gradients.keys():
            gradients[param] = np.atleast_2d(gradients[param])
            self.W[param] = np.atleast_2d(self.W[param])
            for i in range(gradients[param].shape[0]):
                for j in range(gradients[param].shape[1]):
                    self.W[param][i, j] += eps
                    L_prime = self.compute_loss(x, y)
                    self.W[param][i, j] -= eps
                    gradients[param][i, j] = (L_prime-L)/eps

        return gradients

    def fprop(self, x):

        h_a = x.dot(self.W['w1'].T) + self.W['b1']
        h_s = reLU(h_a)
        o_a = h_s.dot(self.W['w2'].T) + self.W['b2']
        o_s = softmax(o_a)

        return dict([('h_a', h_a), ('h_s', h_s), ('o_a', o_a), ('o_s', o_s)])

    def bprop_slow(self, x, y):

        grad_w1, grad_b1, grad_w2, grad_b2 = 0, 0, 0, 0

        for example, label in zip(x, y):

            values = self.fprop(example)

            grad_oa = values['o_s']-onehot(label, self.n_output)
            grad_b2 += grad_oa
            grad_w2 += np.multiply(grad_oa.T, values['h_s'])

            grad_hs = grad_oa.dot(self.W['w2'])
            grad_ha = grad_hs * (values['h_s'] > 0).astype(float)
            grad_w1 += np.multiply(grad_ha.T, example)
            grad_b1 += grad_ha

        return dict([('w1', grad_w1), ('b1', grad_b1),
                     ('w2', grad_w2), ('b2', grad_b2)])

    def bprop_fast(self, x, y):

        values = self.fprop(x)

        grad_oa = values['o_s']-onehot(y, self.n_output)
        grad_b2 = np.sum(grad_oa, axis=0)
        grad_w2 = np.dot(grad_oa.T, values['h_s'])

        grad_hs = np.tensordot(self.W['w2'], grad_oa, (0, 1)).transpose()
        grad_ha = grad_hs * (values['h_s'] > 0).astype(float)
        grad_w1 = np.dot(grad_ha.T, x)
        grad_b1 = np.sum(grad_ha, axis=0)

        return dict([('w1', grad_w1), ('b1', grad_b1),
                     ('w2', grad_w2), ('b2', grad_b2)])

    def set_valid(self, x, y):
        self.valid_data = x
        self.valid_labels = y

    def set_test(self, x, y):
        self.test_data = x
        self.test_labels = y
