import struct

def mapto1hot(input,class_count):
    tmp = [0]*class_count
    tmp[int(input)-1]=1
    return tmp

def import_mnist(dataset_path,labelset_path):

    with open(labelset_path, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = f.read()

    with open(dataset_path, 'rb') as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        raw_data = f.read()


    labels = list(struct.unpack(str(len(labels)) + "B",labels)) 
    raw_data = list(struct.unpack(str(len(raw_data)) + "B",raw_data))

    data = [[0]*cols*rows]*size

    for i in range(size):
        data[i] = raw_data[i * rows * cols : (i + 1) * rows * cols]

    return data, labels