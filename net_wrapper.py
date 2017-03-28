from nn import network
import numpy as np


classes = []
f = open('zip.train')
data = np.loadtxt(f,delimiter=None)
f.close()

for item in data:
    if item[0]==2:
        classes.append([1,0])
    elif item[0]==8:
        classes.append([0,1])

data = data[:,1:]
layers = [256]

'''classes = []
f = open('iris.data')
data = np.loadtxt(f,delimiter=',')
f.close()
for item in data:
    if item[-1]==0.0:
        classes.append([0,0,1])
    elif item[-1]==1.0:
        classes.append([0,1,0])
    else:
         classes.append([1,0,0])

data = data[:,:-1]
layers = [10]'''

perm = np.random.permutation(1273)
train_data = np.array(data)[perm]
classes = np.array(classes)[perm]

train_data = [list(x) for x in train_data]
classes = [list(x) for x in classes]

train_limit = int(round(0.7*len(train_data)))

net = network(train_data[:train_limit],classes[:train_limit],layers,0.2,300)
net.train()
net.testmodel(train_data[train_limit+1:],classes[train_limit+1:],True)
#net.crossvalidate(10,True)