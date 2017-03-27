import numpy as np

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

        #all other layers
        for i in range(2,self.num_layers):

            self.delta[-i] = \
                self.sigmoid_prime(self.z[-i+1][1:]) * \
                    np.dot(self.weights[-i+1][:,1:].T,self.delta[-i+1])

        #update weights
        for i in range(0,self.num_layers-1):
            self.weights[i] -= self.learning_rate *  np.dot(np.matrix(self.delta[i]).T,np.matrix(self.neuron_values[i])) 
     

    def train(self):
        predicted_labels = []
        for i in range(self.epochs):
            for j in range(len(self.train_data)):
                self.feedforward(self.train_data[j])
                self.backpropagate(self.labels[j])

                for k in range(1,len(self.neuron_values[-1])):
                    if self.neuron_values[-1][k] > 0.9:
                        self.neuron_values[-1][k] = 1
                    elif self.neuron_values[-1][k] < 0.1:
                        self.neuron_values[-1][k] = 0
                        
                predicted_labels.append(neuron_values[-1][1:])

            print error(predicted_labels,self.labels)
            
    def error(self,predicted,ref):
        return sum(int(pred==ref) for pred,re in zip(predicted,ref))/len(ref)

    def MSE(self,predicted_labels,ref_labels):
        error = 0
        for pred,ref in zip(predicted_labels,ref_labels):
            error += (pred-ref)**2
        return error/len(predicted_labels)

    def predict(self,input):
        self.feedforward(input)
        print self.neuron_values[-1][1:]

    def crossvalidate(self,folds):
        pass