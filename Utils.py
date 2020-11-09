import numpy as np
import random
import os

def binary_onehot(label, nb_classes):
    onehot_labels = [np.eye(nb_classes, dtype=int)[int(l)] for l in label]
    return np.asarray(onehot_labels)

def normalize(data):
    data = data / np.linalg.norm(data)
    return data

def shuffle(x, y):
    a = list(zip(x, y))
    random.seed(10)
    random.shuffle(a)
    return zip(*a)

def getallfiles(path):

    allfile=[]
    for dirpath,dirnames,filenames in os.walk(path):
        # for dir in dirnames:
        #     allfile.append(os.path.join(dirpath, dir))
        for name in filenames:
            path = os.path.join(dirpath, name)
            n, t = os.path.splitext(path)
            if t == '.wav':
                allfile.append(path)

    return allfile