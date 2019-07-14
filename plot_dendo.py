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

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    labels=labels
)
#plt.show()
plt.savefig('test.png')
print(labels)
print(Z)
