import numpy as np
# todo
# restructure code so that only structural elements on net creation
# add training and testing sets to crossvalidation function
# figure out why MNIST only converges to 8% training error
# implement variable size minibatch
# optimise

class network(object):

    def __init__(self, train_data, labels, layers, learning_rate,\
         epochs,hidden_activation_function = None, \
         output_activation_function = None):

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

        if hidden_activation_function == None:
            print "---- Using sigmoid activation for hidden layers ----"
            self.activation_function = self.sigmoid
            self.activation_function_prime = self.sigmoid_prime

        elif hidden_activation_function == "ReLU":
            print "---- Using Rectified Linear Unit activation for hidden layers ----"
            self.activation_function = self.ReLU
            self.activation_function_prime = self.ReLU_prime

        elif hidden_activation_function == "LReLU":
            print "---- Using Leaky Rectified Linear Unit activation for hidden layers ----"
            self.activation_function = self.LReLU
            self.activation_function_prime = self.LReLU_prime
            self.leakage_parameter = 0.01

        #typical linear output with MSE cost
        if output_activation_function == None:
            print "---- Using linear output with MSE cost ----"
            self.output_activation = (lambda x: x) #linear output function
            self.output_activation_prime = (lambda x: [1] * len(x)) #derivative is 1
            self.delta_output = self.output_delta_MSE
        
        #softmax output with XEntropy cost
        elif output_activation_function == "SoftMax":
            print "---- Using softmax output with XEntropy cost ----"
            self.output_activation = self.softmax
            self.cost_function_prime = self.cross_entropy_for_softmax_prime
            self.delta_output = self.output_delta_sm_xent


    def init_weights(self):
        self.weights = [
                        np.random.rand(self.layers[k+1],self.layers[k]+1)
                        for k in range(len(self.layers)-1)
                       ]

    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self,z):
        sigres = self.sigmoid(z)
        return sigres * (1 - sigres)

    def ReLU(self,z):
        return np.maximum(0,z)

    def ReLU_prime(self,z):
        tmp = np.zeros(len(z))
        tmp[np.greater(z,0)] = 1
        return tmp

    def LReLU(self,z):
        tmp = np.zeros(len(z))
        indices = z > 0
        tmp[indices] = z[indices]
        tmp[~indices] = self.leakage_parameter * z[~indices]
        return tmp

    def LReLU_prime(self,z):
        tmp = np.zeros(len(z))
        indices = z > 0
        tmp[indices] = 1
        tmp[~indices] = self.leakage_parameter
        return tmp

    def cross_entropy_for_softmax_prime(self,z,labels):
        return z - labels

    def softmax(self,z):
        stable = np.exp(z - np.max(z)) #to avoid NaNs
        return stable / np.sum(stable)

    def MSE_cost_prime(self,z,label):
        return z - label

    def output_delta_MSE(self,neuron_vals,targets,der_output):
        return self.MSE_cost_prime(neuron_vals,targets) * self.output_activation_prime(der_output)

    def output_delta_sm_xent(self,neuron_vals,targets,der_output):
        return self.cross_entropy_for_softmax_prime(neuron_vals,targets)

    def feedforward(self, input):
        self.neuron_values[0][1:] = input

        for i in range(1,self.num_layers-1):
            self.z[i] = np.dot(self.weights[i-1],self.neuron_values[i-1])
            self.neuron_values[i][1:] = self.activation_function(self.z[i])
        
        self.z[-1] = np.dot(self.weights[-1],self.neuron_values[-2])
        self.neuron_values[-1][1:] = self.output_activation(self.z[-1])

    def backpropagate(self,label):
        
        #output layer
        self.delta[ -1 ] = self.delta_output(self.neuron_values[ -1 ][1:],label,self.z[ -1 ])
        
        #all other layers
        for i in range(2,self.num_layers):
            self.delta[-i] = \
                self.activation_function_prime(self.z[-i]) * \
                    np.dot(self.weights[-i+1][:,1:].T,self.delta[-i+1])

        #update weights
        for i in range(0,self.num_layers-1):
            self.weights[i] -= self.learning_rate *  np.dot(np.matrix(self.delta[i]).T,np.matrix(self.neuron_values[i])) 
     

    def train(self):

        last_err = 1
        self.learning_rate = self.cv_starting_rate
        old_weights = []
        print 'training....'
        for i in range(self.epochs):
            train_error = self.testmodel(self.train_data,self.labels,True)
            if train_error <= last_err:
                self.learning_rate *= 1.1
                old_weights = list(self.weights)
            else:
                self.learning_rate *= 0.5
                self.weights = list(old_weights)
            last_err = train_error

            print("current epoch: " + str(i) + "/" + str(self.epochs) + " train error: " + str(train_error) + " learning rate: " + str(self.learning_rate))

            '''if i%(round(self.epochs/10))==0:
                
                print("current epoch: " + str(i) + "/" + str(self.epochs) + " train error: " + str(train_error) + " learning rate: " + str(self.learning_rate))
            '''
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