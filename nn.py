import numpy as np
from random import uniform


class network(object):

    def __init__(self, train_data, labels, layers,learning_rate, epochs):
        self.train_data = train_data
        self.labels = labels
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_layers = len(self.layers)

        self.weights = [
                        np.random.rand(layers[k+1],layers[k]+1)
                        for k in range(len(layers)-1)
                       ]

        self.neuron_values = [np.ones(n+1) for n in layers]
        self.z = [np.ones(n+1) for n in layers]
        self.delta = [np.ones(n) for n in layers[1:]]
        
        '''print('weights')
        print(self.weights)
        print('neuron values')
        print(self.neuron_values)
        print "layers"
        print self.layers
        print('\n')'''

        
    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def feedforward(self, input):
        self.neuron_values[0] = np.array([1] + input)

        for i in range(1,self.num_layers): #layer
            self.z[i][1:] = np.dot(self.weights[i-1],self.neuron_values[i-1])
            self.neuron_values[i][1:] = self.sigmoid(self.z[i][1:])
        

    def backpropagate(self,label):
         #output layer
        self.delta[ -1 ] = (self.neuron_values[ -1 ][1:] - label) * \
                                self.sigmoid_prime(self.z[ -1 ][1:])
            
        '''print "deltas"
        print self.delta'''

        #all other layers
        for i in range(2,self.num_layers):
            '''print "cur weights"
            print self.weights[-i+1][:,1:].T
            print"cur delta"
            print self.delta[-i+1]
            print "cur z"
            print self.z[-i][1:]'''

            self.delta[-i] = \
                self.sigmoid_prime(self.z[-i+1][1:]) * \
                    np.dot(self.weights[-i+1][:,1:].T,self.delta[-i+1])

        '''print '\n'
        print "weight update"
        print "deltas"
        print self.delta
        print "neuron vals"
        print self.neuron_values'''
        #update weights
        for i in range(0,self.num_layers-1):
            '''print"neuron vals"
            print np.matrix(self.neuron_values[i]).T
            print "cur deltas"
            print np.matrix(self.delta[i])
            print "cur weights"
            print self.weights[i]
            print "res"
            print np.dot(np.matrix(self.delta[i]).T,np.matrix(self.neuron_values[i]))''' 

            self.weights[i] -= self.learning_rate *  np.dot(np.matrix(self.delta[i]).T,np.matrix(self.neuron_values[i])) 
     

    def train(self):
        for i in range(self.epochs):
            for j in range(len(self.train_data)):
                self.feedforward(self.train_data[j])
                self.backpropagate(self.labels[j])

    def predict(self,input):
        self.feedforward(input)
        print self.neuron_values[-1][1:]

    def crossvalidate(self,folds):
        pass





train_data = []
f = open('zip.train')
for line in f:
    curline = line.split(' ')[:-2]
    if (float(curline[0])==2.0) | (float(curline[0])==8.0):
        train_data.append([float(x) for x in curline])
f.close()
for data in train_data:
    if data[0]==2:
        data[0]=0
    elif data[0]==8:
        data[0]=1

train_data = np.array(train_data)
labels = train_data[:,0]
print labels
train_data = train_data[:,1:]
layers = [256,256,1]

'''train_data = [[0,0],[0,1],[1,0],[1,1]]
labels = [0,1,1,0]
layers = [2,2,1]'''

net = network(train_data,labels,layers,0.7,1000)
net.train()
#net.feedforward(train_data[0])
#net.backpropagate(labels[0])
net.predict(train_data[0])
print labels[0]
net.predict(train_data[1])
print labels[1]
net.predict(train_data[2])
print labels[2]
net.predict(train_data[3])
print labels[3]
'''print "neuron vals"
print net.neuron_values
print "delta"
print net.delta'''
