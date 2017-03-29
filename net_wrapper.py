from nn import network
import numpy as np
from utils import import_mnist,mapto1hot

'''f = open('ml_mnist_data/zip.train')
data = np.loadtxt(f)
f.close()

classes = [mapto1hot(x,10) for x in data[:,0]]
train_data = data[:,1:]
layers = [32]'''

'''train_data,classes = import_mnist('mnist_data/train.idx3-ubyte','mnist_data/train_labels.idx1-ubyte')
classes = [mapto1hot(x,10) for x in classes]
train_data = list(np.array(train_data) / 255)
layers = [100]'''

f = open('iris.data')
data = np.loadtxt(f,delimiter=',')
f.close()
classes = [mapto1hot(x,3) for x in data[:,-1]]
train_data = data[:,:-1]
layers = [10]

net = network(train_data,classes,layers,0.2,100)
net.crossvalidate(10,True)