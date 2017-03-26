import numpy as np
from random import uniform


class network(object):

    def __init__(self, train_data, labels, layers,learning_rate, epochs):
        self.train_data = train_data
        self.labels = labels
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.weights = [
                        np.random.rand(layers[k+1],layers[k]+1)
                        for k in range(len(layers)-1)
                       ]

        self.neuron_values = [np.ones(n+1) for n in layers]
        self.befactiv = [np.ones(n+1) for n in layers]
        self.delta = [np.ones(n+1) for n in layers]
        
        print('weights')
        print(self.weights)
        print('neuron values')
        print(self.neuron_values)
        print('\n')

        
    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self,z):
        return z * (1 - z)

    
    def feedforward(self, input):

        self.neuron_values[0] = np.array([1] + input)

        for i in range(1,len(self.layers)): #layer
            self.befactiv[i][1:] = np.dot(self.weights[i-1],self.neuron_values[i-1])
            self.neuron_values[i][1:] = self.sigmoid(self.befactiv[i][1:])
        

    def backpropagate(self,label):
         #output layer
        self.delta[len(self.layers)-1][1:] = \
            (self.neuron_values[ len(self.layers)-1 ][1:] - label) \
                * self.sigmoid_prime(self.neuron_values[ len(self.layers)-1 ][1:])
        #print "deltas"
        #print self.delta
        #print "neuron vals"
        #print self.neuron_values

        #all other layers
        for i in range(len(self.layers)-2,-1,-1):
            self.delta[i] = self.sigmoid_prime(self.befactiv[i]) * np.dot(self.weights[i].T,self.delta[i+1][1:])

        #print self.weights
        #print self.neuron_values
        #print self.delta

        #update weights
        for i in range(len(self.layers)-1):
            self.weights[i] -= self.learning_rate * self.neuron_values[i] * self.delta[i]
        

    def train(self):
        for i in range(self.epochs):
            for j in range(len(self.train_data)):
                self.feedforward(self.train_data[j])
                self.backpropagate(self.labels[j])

    def predict(self):
        pass

    def crossvalidate(self,folds):
        pass







train_data = [[0,0],[0,1],[1,0],[1,1]]
labels = [-1,-1,-1,1]
layers = [3,3,1]
net = network(train_data,labels,layers,0.7,1000)
net.train()
#net.feedforward(train_data[0])
#net.backpropagate(labels[0])
print net.neuron_values
