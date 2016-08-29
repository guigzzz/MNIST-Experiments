import numpy as np
import random


class network(object):

    def __init__(self, training_data, checking_data, layers, epochs, learning_rate, biases, weights):
        self.training_data = training_data
        self.checking_data = checking_data
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.biases = biases
        self.weights = weights
        self.all_outputs = []

    def feedforward(self, data, layer_count):
        
        if layer_count == 1:
            data = self.training_data

        if layer_count != len(self.layers): #check for base case, e.g when calculated output values
            outputs = []
            for neuron in range(self.layers[layer_count]):
                output = sigmoid(sum(data[i]*self.weights[(layer_count-1)*4 + 2*neuron +i] for i in range(self.layers[layer_count-1])) + self.biases[layer_count-1])
                outputs.append(output)

            self.all_outputs.append(outputs)
            return self.feedforward(outputs,layer_count+1)

        return data


    def backprop(self): # dE/dw = -(target(o1)-out(o1)) * out(o1)*(1-out(o1))*out(h1)
        tmp_weights = []

        """
        for weight in reversed(self.weights):
            

            new_weight = weight - self.learning_rate * dEdw
            tmp_weights.append(new_weight)

        return tmp_weights
        """



    
    def run(self):
        error = 0
        for i in range(1): #, self.epochs


            outputs = self.feedforward(self.training_data, 1)
            #print(outputs)
            error = sum(0.5*np.square(self.checking_data[i]-outputs[i]) for i in range(len(outputs)))
            self.weights = self.backprop()
            print('epoch: ' + str(i+1) + ', error: ' + str(error))
            print(self.all_outputs)

            self.all_outputs = []
    


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return z * (1 - z)
