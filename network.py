import numpy as np
import random


class network(object):

    def __init__(self, training_data, checking_data, layers, epochs, learning_rate, biases, weights):
        self.training_data = training_data
        self.checking_data = checking_data
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.biases = biases #add random generation
        self.weights = weights
        self.all_outputs = []

    def feedforward(self, data, layer_count):
        
        if layer_count == 1:
            data = self.training_data

        if layer_count != len(self.layers): #check for base case, e.g when calculated output values
            outputs = []
            for neuron in range(self.layers[layer_count]):
                output = sigmoid(sum(data[i]*self.weights[(layer_count-1)][2*neuron +i] for i in range(self.layers[layer_count-1])) + self.biases[layer_count-1])
                outputs.append(output)

            self.all_outputs.append(outputs)
            return self.feedforward(outputs,layer_count+1)

        return data


    def backprop(self): # dE/dw = -(target(o1)-out(o1)) * out(o1)*(1-out(o1))*out(h1) , out(o1)*(1-out(o1)) = sigmoid_prime(out(o1))

        tmp_weights = []
        self.all_outputs = [self.training_data] + self.all_outputs



        #do network output layer




        delta = []
        deltah = []

        for i in range(self.layers[-1]):
            delta.append(-(self.checking_data[i]-self.all_outputs[-1][i]) * sigmoid_prime(self.all_outputs[-1][i]))
        

        neuron_count_previous = self.layers[-2]



        for i in range(len(self.weights[-1])/2):
            discard_delta = delta[i]
            for j in range(neuron_count_previous):
                tmp_weights.append(self.weights[-1][2*i+j] - self.learning_rate*discard_delta*self.all_outputs[-2][j])




        #do the rest




        for i in range(self.layers[-2]):

            delta_tmp = sum(delta[k]*self.weights[-1][k] for k in range(self.layers[-2]))

            deltah.append(delta_tmp*sigmoid_prime(self.all_outputs[-2][i])) #check sum


        for i in range(len(self.weights[-2])/2):
            discard_delta = deltah[i]
            for j in range(neuron_count_previous): 
                tmp_weights.append(self.weights[-2][2*i+j] - self.learning_rate*discard_delta*self.all_outputs[-3][j])





        chunksize = 4
        self.weights = list(reversed([tmp_weights[i:i + chunksize] for i in xrange(0, len(tmp_weights), chunksize)]))



    
    def run(self):
        error = 0

        for i in range(1, self.epochs+1): #, self.epochs
            outputs = self.feedforward(self.training_data, 1)
            #print(outputs)
            error = sum(0.5*np.square(self.checking_data[i]-outputs[i]) for i in range(len(outputs)))

            """

            print('initial weights')
            print(self.weights)
            print('')

            """

            self.backprop()

            if i % 10000 == 0:
                print(('epoch: ' + str(i) + ', MSE: ' + str(error)) )

            """

            print('')
            print('all_outputs')
            print(self.all_outputs)
            print('')
            print('weights')
            print(self.weights)
            print('')

            """

            #print([self.training_data] + self.all_outputs)

            self.all_outputs = [] #reset outputs

        print('outputs at final epoch:')
        print('')
        print(outputs)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return z * (1 - z)
