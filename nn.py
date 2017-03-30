import numpy as np

class network(object):

    def __init__(self, train_data, labels, layers, learning_rate, epochs):
        self.train_data = train_data

        input_size = len(train_data[0])
        if type(labels[0]) == list:
            output_size = len(labels[0])
        else:
            output_size = 1

        self.labels = labels
        self.layers = np.array([input_size] + layers + [output_size])
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.cv_starting_rate = learning_rate
        self.num_layers = len(self.layers)

        self.init_weights()

        self.neuron_values = [np.ones(n+1) for n in self.layers]
        self.z = [np.ones(n) for n in self.layers]
        self.delta = [np.ones(n) for n in self.layers[1:]]

    def init_weights(self):
        self.weights = [
                        np.random.rand(self.layers[k+1],self.layers[k]+1)
                        for k in range(len(self.layers)-1)
                       ]

    def tanh(self,z):
        return self.sigmoid(2*z)-self.sigmoid(-2*z)
    
    def tanh_prime(self,z):
        sigres = self.sigmoid(z)
        return 1 - sigres*sigres

    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self,z):
        sigres = self.sigmoid(z)
        return sigres * (1 - sigres)

    def cross_entropy_cost_prime(z,label):
        return (z - label)/(z * (1 - z))

    def MSE_cost_prime(z,label):
        return z - label
    
    def feedforward(self, input):

        self.neuron_values[0][1:] = input

        for i in range(1,self.num_layers): #layer

            self.z[i] = np.dot(self.weights[i-1],self.neuron_values[i-1])
            self.neuron_values[i][1:] = self.sigmoid(self.z[i])
        

    def backpropagate(self,label,Output_activation = MSE_cost_prime):
        #output layer
        self.delta[ -1 ] = Output_activation(self.neuron_values[ -1 ][1:],label) * \
                                self.sigmoid_prime(self.z[ -1 ])

        #all other layers
        for i in range(2,self.num_layers):

            self.delta[-i] = \
                self.sigmoid_prime(self.z[-i]) * \
                    np.dot(self.weights[-i+1][:,1:].T,self.delta[-i+1])

        #update weights
        for i in range(0,self.num_layers-1):
            self.weights[i] -= self.learning_rate *  np.dot(np.matrix(self.delta[i]).T,np.matrix(self.neuron_values[i])) 
     

    def train(self):
        '''last_err = 1
        self.learning_rate = self.cv_starting_rate
        old_weights = []'''
        print 'training....'
        for i in range(self.epochs):
            train_error = self.testmodel(self.train_data,self.labels,True)
            '''if train_error <= last_err:
                self.learning_rate *= 1.1
                old_weights = list(self.weights)
            else:
                self.learning_rate *= 0.5
                self.weights = list(old_weights)
            last_err = train_error'''

            if i%(round(self.epochs/10))==0:
                
                print("current epoch: " + str(i) + "/" + str(self.epochs) + " train error: " + str(train_error) + " learning rate: " + str(self.learning_rate))

            for j in range(len(self.train_data)):
                
                self.feedforward(self.train_data[j])
                self.backpropagate(self.labels[j])
            
    def error(self,predicted,ref):
        errorlst = [int(list(pred)!=list(re)) for pred,re in zip(predicted,ref)]
        return float(sum(errorlst))/len(errorlst)

    def MSE(self,predicted_labels,ref_labels):
        error = 0
        for pred,ref in zip(predicted_labels,ref_labels):
            error += (pred[0]-ref)**2
        return error/len(predicted_labels)

    def predict(self,input):
        output = []
        for datapoint in input:
            self.feedforward(datapoint)
            output.append(list(self.neuron_values[-1][1:]))
        return output

    def roundclasses(self,input):
        output = []
        for i in range(len(input)):
            output.append([int(round(x)) for x in input[i]])
        return output

    def testmodel(self,test_set,test_classes,classround = False):
        if classround:
            predictions = self.roundclasses(self.predict(test_set))
            return self.error(predictions,test_classes)
        else:
            predictions = self.predict(test_set)
            return self.error(predictions,test_classes)

    def crossvalidate(self,folds,randomise = False):
        if randomise:
            from random import shuffle
            items = list(self.train_data)
            perm = np.random.permutation(len(self.train_data))
            self.train_data = np.array(self.train_data)[perm]
            self.labels = np.array(self.labels)[perm]

        train_limit = int(round(0.7*len(self.train_data)))

        test_data = self.train_data[train_limit+1:]
        self.train_data = self.train_data[:train_limit]
        test_labels = self.labels[train_limit+1:]
        self.labels = self.labels[:train_limit]

        slices = [self.train_data[i::folds] for i in range(folds)]
        label_slices = [self.labels[i::folds] for i in range(folds)]

        best_weights = []
        best_error = 1

        for k in range(len(slices)):
            self.train_data = np.concatenate(np.array(slices)[np.arange(len(slices))!=k])
            #self.train_data = np.array(self.train_data)

            self.labels = np.concatenate(np.array(label_slices)[np.arange(len(slices))!=k])
            validation = slices[k]
            validation_labels = label_slices[k]

            self.train()
            err = self.testmodel(validation,validation_labels,True)
            print "validation error: " + str(err)

            if err<best_error:
                best_error = err
                best_weights = list(self.weights)

            self.init_weights()
        
        self.weights = best_weights

        te = self.testmodel(test_data,test_labels,True)
        print "test error: " + str(te)