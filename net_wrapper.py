import network
import numpy as np

training_data = [0.05,0.1]

checking_data = [0.01,0.99]

layers = [2, 2, 2]

epoch = 10000

learning_rate = 0.5

biases = [0.35, 0.6]

weights = [0.15, 0.25, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55]

#initialise network

net = network.network(training_data, checking_data, layers, epoch,
                      learning_rate, biases, weights)

net.run()




