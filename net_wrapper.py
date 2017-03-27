from nn import network
import numpy as np

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