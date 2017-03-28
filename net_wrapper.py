from nn import network
import numpy as np

'''train_data = []
f = open('zip.train')
for line in f:
    curline = line.split(' ')[:-1]
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
train_data = train_data[:,1:]
layers = [256]'''

data = []
classes = []
f = open('iris.data')
for line in f:
    curline = line.split('\n')[0].split(',')
    data.append([float(x) for x in curline[:-1]])
    if float(curline[-1])==0.0:
        classes.append([0,0,1])
    elif float(curline[-1])==1.0:
        classes.append([0,1,0])
    else:
         classes.append([1,0,0])
f.close()
layers = [10]

perm = np.random.permutation(len(data))
train_data = np.array(data)[perm]
classes = np.array(classes)[perm]

train_data = [list(x) for x in train_data]
classes = [list(x) for x in classes]

train_limit = int(round(0.7*len(train_data)))

net = network(train_data[:train_limit],classes[:train_limit],layers,0.2,150)
net.train()
net.testmodel(train_data[train_limit+1:],classes[train_limit+1:],True)
#net.crossvalidate(10,True)