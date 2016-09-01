import numpy as np
from random import uniform


class network(object):

    def __init__(self, training_data, checking_data, layers, epochs, learning_rate, net_type):
        self.training_data = training_data
        self.checking_data = checking_data
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        # self.biases = biases
        # self.weights = weights  # add random generation

        self.biases = [uniform(0, 1) for y in layers[1:]]

        self.number_of_weights = []
        for i in range(len(layers) - 1):
            self.number_of_weights.append(layers[i] * layers[i + 1])

        self.weights = []
        for k in range(len(self.number_of_weights)):
            self.weights.append([uniform(0, 1)
                                 for i in range(self.number_of_weights[k])])

        print('biases:')
        print(self.biases)
        print('weights')
        print(self.weights)
        print('\n')

        self.number_of_weights = list(reversed(self.number_of_weights))

        self.all_outputs = []
        self.epoch = 0
        self.net_type = net_type

    def div_to_chunks(self, tmp_weights):

        #print(tmp_weights)

        index = 0
        self.weights = []
        for i in range(len(self.number_of_weights)):
            self.weights.append(
                tmp_weights[index:index + self.number_of_weights[i]])
            index += self.number_of_weights[i]

        self.weights = list(reversed(self.weights))

        #print(self.weights)

    def feedforward(self, data, layer_count, starting_data):

        if layer_count == 1:
            data = starting_data

        if layer_count != len(self.layers):
            outputs = []
            for neuron in range(self.layers[layer_count]):
                output = sigmoid(sum(data[i] * self.weights[(layer_count - 1)][self.layers[layer_count - 1] * neuron + i]
                                     for i in range(self.layers[layer_count - 1])) + self.biases[layer_count - 1])
                outputs.append(output)

            self.all_outputs.append(outputs)
            return self.feedforward(outputs, layer_count + 1, starting_data)

        return data

    def backprop(self):

        tmp_weights = []

        # do network output layer

        delta = []
        deltah = []

        for i in range(self.layers[-1]):
            delta.append(-(self.checking_data[i] - self.all_outputs[-1][
                         i]) * sigmoid_prime(self.all_outputs[-1][i]))

        for i in range(self.layers[-1]):
            discard_delta = delta[i]
            for j in range(self.layers[-2]):
                tmp_weights.append(
                    self.weights[-1][self.layers[-2] * i + j] - self.learning_rate * discard_delta * self.all_outputs[-2][j])

        # do the other layers, going through network backwards

        for l in range(2, len(self.layers)):

            for i in range(self.layers[-l]):

                delta_tmp = sum(delta[k] * self.weights[-l + 1][k * self.layers[-l + 1] + i]
                                for k in range(self.layers[-l + 1]))  # seg fault here

                deltah.append(
                    delta_tmp * sigmoid_prime(self.all_outputs[-l][i]))

            #print(self.all_outputs)

            for i in range(self.layers[-l]):

                discard_delta = deltah[i]
                for j in range(self.layers[-l - 1]):
                    tmp_weights.append(
                        self.weights[-l][self.layers[- l - 1] * i + j] - self.learning_rate * discard_delta * self.all_outputs[-l - 1][j])

            delta = deltah

        self.div_to_chunks(tmp_weights)

    def run(self):
        error = 0

        for self.epoch in range(1, self.epochs + 1):  # , self.epochs

            if self.net_type == 'gate':
                outputs = self.feedforward([], 1, self.training_data[
                    (self.epoch - 1) % 4])
                error = sum(
                    0.5 * np.square(self.checking_data[(self.epoch - 1) % 4] - outputs[k]) for k in range(len(outputs)))
                self.all_outputs = [self.training_data[
                    (self.epoch - 1) % 4]] + self.all_outputs

            elif self.net_type == 'test':
                outputs = self.feedforward([], 1, self.training_data)
                error = sum(
                    0.5 * np.square(self.checking_data[k] - outputs[k]) for k in range(len(outputs)))
                self.all_outputs = [self.training_data] + self.all_outputs

            self.backprop()

            if self.epoch % (self.epochs / 10) == 0:
                print(('epoch: ' + str(self.epoch) + ', MSE: ' +
                       str(error)))  # + ', outputs: ' + str(outputs)

            self.all_outputs = []  # reset outputs

        print('')
        print('final outputs:')
        if self.net_type == 'gate':
            for i in range(len(self.training_data)):
                print('inputs: ' + str(self.training_data[i]) + ", output: " + str(
                    self.feedforward([], 1, self.training_data[i])))

        elif self.net_type == 'test':
            print('inputs: ' + str(self.training_data) + ", output: " + str(
                self.feedforward([], 1, self.training_data)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return z * (1 - z)
