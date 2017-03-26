import numpy as np
from random import uniform


class network(object):

    def __init__(self, train_data, labels, layers, epochs):
        self.train_data = train_data
        self.labels = labels
        self.layers = layers
        self.epochs = epochs

        self.weights = [
                        np.random.rand(layers[k+1],layers[k]+1)
                        for k in range(len(layers)-1)
                       ]

        self.neuron_values = [np.ones(n+1) for n in layers]
        self.befactiv = self.neuron_values

        print('weights')
        print(self.weights)
        print('neuron values')
        print(self.neuron_values)
        print('\n')

        self.delta = [np.ones(n) for n in layers]

    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self,z):
        return z * (1 - z)

    
    def feedforward(self, input):

        self.neuron_values[0] = np.array([1] + input)

        for i in range(1,len(self.layers)): #layer
            self.befactiv[i][1:] = np.dot(self.weights[i-1],self.neuron_values[i-1])
            self.neuron_values[i][1:] = self.sigmoid(self.befactiv[i][1:])

        

    def backpropagate(self):
         #output layer
        self.delta[len(self.layers)-1] = \
            (self.neuron_values[ len(self.layers)-1 ][1:] - self.labels[0]) \
                * sigmoid_prime(self.neuron_values[ len(self.layers)-1 ][1:])

        #all other layers
        for i in range(len(self.layers)-2,0,-1):
            self.delta[i] = self.sigmoid_prime(self.befactiv[i]) * dot(self.delta[i+1],self.weights[i])

        #update weights
        

    def train(self):
        self.feedforward(self.train_data[0])






train_data = [[1,2,3],[3,2,1]]
labels = [1,-1]
layers = [3,3,1]
net = network(train_data,labels,layers,1000)
net.train()
print net.neuron_values
