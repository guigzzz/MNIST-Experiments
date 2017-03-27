import numpy as np

class network(object):

    def __init__(self, train_data, labels, layers,learning_rate, epochs):
        self.train_data = train_data

        input_size = len(train_data[0])
        if type(labels[0])==list:
            output_size=len(labels[0])
        else:
            output_size=1

        self.labels = labels
        self.layers = [input_size] + layers + [output_size]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_layers = len(self.layers)

        self.weights = [
                        np.random.rand(self.layers[k+1],self.layers[k]+1)
                        for k in range(len(self.layers)-1)
                       ]

        self.neuron_values = [np.ones(n+1) for n in self.layers]
        self.z = [np.ones(n+1) for n in self.layers]
        self.delta = [np.ones(n) for n in self.layers[1:]]

        
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
                self.sigmoid_prime(self.z[-i][1:]) * \
                    np.dot(self.weights[-i+1][:,1:].T,self.delta[-i+1])

        #update weights
        for i in range(0,self.num_layers-1):
            self.weights[i] -= self.learning_rate *  np.dot(np.matrix(self.delta[i]).T,np.matrix(self.neuron_values[i])) 
     

    def train(self):
        print 'training....'
        for i in range(self.epochs):

            if i%(round(self.epochs/10))==0:
                print("current epoch: " + str(i))

            for j in range(len(self.train_data)):
                self.feedforward(self.train_data[j])
                self.backpropagate(self.labels[j])
            
    def error(self,predicted,ref):
        return sum(int(pred!=re) for pred,re in zip(predicted,ref))/len(ref)

    def MSE(self,predicted_labels,ref_labels):
        error = 0
        for pred,ref in zip(predicted_labels,ref_labels):
            error += (pred[0]-ref)**2
        return error/len(predicted_labels)

    def predict(self,input):
        output = []
        for datapoint in input:
            self.feedforward(datapoint)
            #print self.neuron_values
            
            output.append(list(self.neuron_values[-1][1:]))
        return output

    def roundclasses(self,input):
        output = []
        for i in range(len(input)):
            output.append([int(round(x)) for x in input[i]])
        return output
        

    def crossvalidate(self,folds,randomise=False):
        if randomise:
            from random import shuffle
            items = list(self.train_data)
            shuffle(self.train_data)

        slices = [self.train_data[i::folds] for i in range(folds)]

        for sub in slices:
            self.train_data = []
            self.train_data = [self.train_data + sl for sl in slices if sl!=sub]
            validation = sub

            print "train data"
            print self.train_data
            print "val"
            print validation
            self.train()
            predictions = self.predict(validation)
            predictions = self.roundclasses(predictions)
            print "validation error: " + str(self.error(predictions,self.labels))
