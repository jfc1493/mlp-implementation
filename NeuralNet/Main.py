'''
Created on Nov 27, 2017

@author: jean-francis
'''

from neuralnetwork import NN
from functions import plot_decision
import numpy as np
import time
from matplotlib import pyplot as plt
import gzip,pickle


np.set_printoptions(suppress=True)



class moon:
    
    def __init__(self):
        dataset = np.loadtxt('/home/jean-francis/workspace/IFT6390_Devoir2/2moons.txt')
        data = dataset[:,:-1]
        labels = dataset[:,-1]

        self.train_data=data[:700]
        self.valid_data=data[700:900]
        self.test_data=data[900:1100]
        
        self.train_labels=labels[:700]
        self.valid_labels=labels[700:900]
        self.test_labels=labels[900:1100]
        
        self.n_features=data.shape[1]
        self.n_categories=np.unique(self.train_labels).shape[0]

class mnist:
    
    def __init__(self):
        f=gzip.open('/home/jean-francis/workspace/IFT6390_Devoir2/mnist.pkl.gz')
        dataset=pickle.load(f)
    
        self.train_data=dataset[0][0]
        self.valid_data=dataset[1][0]  
        self.test_data=dataset[2][0]
        
        self.train_labels=dataset[0][1]
        self.valid_labels=dataset[1][1]
        self.test_labels=dataset[2][1]
            
        self.n_features=self.train_data.shape[1]
        self.n_categories=np.unique(self.train_labels).shape[0]
    
    

if __name__ == '__main__':
    


    
    moon=moon()
    
    #1 et 2
    
    nn = NN(2,4,2)
    
    normal=nn.bprop_slow(moon.train_data[:1], moon.train_labels[:1])
    numerical=nn.FD_gradients(moon.train_data[:1], moon.train_labels[:1])
    
    print("\n" "-----Gradient par les formules-----" + "\n")
    print(sorted(normal.items()))
    print("\n" + "-----Gradient par difference finie-----" + "\n")
    print(sorted(numerical.items()))

    
    #3 et 4
    
    normal=nn.bprop_slow(moon.train_data[:10], moon.train_labels[:10])
    numerical=nn.FD_gradients(moon.train_data[:10], moon.train_labels[:10])
    
    print("\n" "-----Gradient par les formules-----" + "\n")
    print(sorted(normal.items()))
    print("\n" + "-----Gradient par difference finie-----" + "\n")
    print(sorted(numerical.items()))
    

    #5
    
    hidden_units=[2,3,8]
    weight_decay=[0.0001,0.005,0.1]
    epochs=[1,4,25]
    
    
    # Effet du nombre d'epoch #
    for n in epochs:
        nn = NN(2,8,2)
        nn.fit(moon.train_data, moon.train_labels, epochs=n)
        print("Number of epochs: " + str(n))
        plot_decision(moon.valid_data, moon.valid_labels, nn)
        plt.show()
    
    
    # Effet du weight decay #
    for wd in weight_decay:
        nn = NN(2,8,2)
        nn.fit(moon.train_data, moon.train_labels,wd=wd)
        print("Weight decay: " + str(wd))
        plot_decision(moon.valid_data, moon.valid_labels, nn)
        plt.show()
        
        
    # Effet du nombre de neurones caches #
    for units in hidden_units:
        nn = NN(2,units,2)
        nn.fit(moon.train_data, moon.train_labels)
        print("Hidden units: " + str(units))
        plot_decision(moon.valid_data, moon.valid_labels, nn)
        plt.show()
    
    

    
    #6 et #7
    
    nn=NN(2,4,2)
    
    w_loop=nn.bprop_slow(moon.train_data[:1], moon.train_labels[:1])
    w_matrix=nn.bprop_fast(moon.train_data[:1], moon.train_labels[:1])
    
    print("\n" "-----Gradient avec boucle-----" + "\n")
    print(sorted(w_loop.items()))
    print("\n" + "-----Gradient avec matrices-----" + "\n")
    print(sorted(w_matrix.items()))
    
    w_loop=nn.bprop_slow(moon.train_data[:10], moon.train_labels[:10])
    w_matrix=nn.bprop_fast(moon.train_data[:10], moon.train_labels[:10])
    
    print("\n" "-----Gradient avec boucle-----" + "\n")
    print(sorted(w_loop.items()))
    print("\n" + "-----Gradient avec matrices-----" + "\n")
    print(sorted(w_matrix.items()))

    
    
    #8
    
    mnist=mnist()
    
    nn=NN(mnist.n_features,128,mnist.n_categories)
    
    print("-----Temps d'entrainement pour un epoch-----")
    
    start=time.time()
    nn.fit(mnist.train_data, mnist.train_labels, K=100, epochs=1,fast=False)# 1 epoch et backprop avec boucle (fast=false)
    stop=time.time()
    print("Avec boucle: " + str(stop-start) + "s")
    
    start=time.time()
    nn.fit(mnist.train_data, mnist.train_labels, K=100, epochs=1,fast=True)# 1 epoch et backprop avec calcul matriciel
    stop=time.time()
    print("Avec calcul matriciel: " + str(stop-start) +  "s")
    

    
    #9 et #10
    
    mnist=mnist()
    
    nn=NN(mnist.n_features,128,mnist.n_categories)
    nn.set_valid(mnist.valid_data,mnist.valid_labels)
    nn.set_test(mnist.test_data,mnist.test_labels)
    nn.fit(mnist.train_data,mnist.train_labels,K=128, epochs=25,fast=True,lrate=0.005,wd=0.000015,print_stats=True)




    
    

    