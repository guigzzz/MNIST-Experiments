from nn import network
import numpy as np

def mapto1hot(input,class_count):
    tmp = [0]*class_count
    tmp[int(input)-1]=1
    return tmp

f = open('zip.train')
data = np.loadtxt(f)
f.close()

classes = [mapto1hot(x,10) for x in data[:,0]]
train_data = data[:,1:]
layers = [32]

'''f = open('iris.data')
data = np.loadtxt(f,delimiter=',')
f.close()
classes = [mapto1hot(x,3) for x in data[:,-1]]
train_data = data[:,:-1]
layers = [10]'''

net = network(train_data,classes,layers,3,100)
net.crossvalidate(10,True)