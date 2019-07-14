import csv
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.spatial.distance import euclidean
configs = []
with open('All_blocklocations.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = -1;
    for row in csv_reader:
        i+=1
        if (i==0):
            continue
        configs.append([row[5],float(row[1]),float(row[2])])

batch = configs[0:15]
X = [x[1:] for x in batch]
labels = [x[0] for x in batch]

Z = linkage(X, 'average')
c, coph_dists = cophenet(Z, pdist(X))
print(c)


def get_linkage(id):
    batch = configs[16*id:16*(id+1)-1]
    X = [x[1:] for x in batch]
    labels = [x[0] for x in batch]

    Z = linkage(X, 'average')
    c, coph_dists = cophenet(Z, pdist(X))
    print(c)
    return Z,X,labels


def break_program_to_subprograms(program):
    

