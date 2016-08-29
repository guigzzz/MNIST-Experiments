import network
import numpy as np

 #network learns to map 0.05 to 0.01 and 0.1 to 0.99

training_data = [0.05,0.1]

checking_data = [0.01,0.99]

layers = [2, 2, 2]

epochs = 10000

learning_rate = 0.5

biases = [0.35, 0.6]

weights = [[0.15, 0.2, 0.25, 0.3], [0.4, 0.45, 0.5, 0.55]]

#initialise network

net = network.network(training_data, checking_data, layers, epochs,
                      learning_rate, biases, weights)

#run network

net.run()
