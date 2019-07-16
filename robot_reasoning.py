import csv
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.spatial.distance import euclidean
configs = []
COLOURS = {"blue":"b","green":"g","yellow":"y","red":"r","orange":"r","brown":"r"}
subscenes = {}
with open('All_blocklocations.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = -1;
    for row in csv_reader:
        i+=1
        if (i==0):
            continue
        configs.append([row[5],float(row[1]),float(row[2])])

batch = configs[0:15]
print(batch)
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
    subprograms = []
    program = program.split(' ')
    subs = []
    for pr in program:
        if "arg" in pr:
            subs.append(pr)
            subprograms.append(subs)
            subs = []
        elif "ans" in pr:
            print("df`")
            subs.append(pr)
            subprograms.append(subs)
            subs = []
        else:
            subs.append(pr)
    return subprograms
def find_clusters(colour,num,subscene):
    #print(colour,num)
    outscene = []
    if (subscene == None):
        if (num==1):
            print(num,"sdsd")
            for x in labels:
                if colour in x:
                    outscene.append(1)
                else:
                    outscene.append(0)
    return outscene   
def execute_subprogram(subprogram,subscene):
    if (subprogram[0]=="clusterof"):
        if (subprogram[1]=="all"):
            return find_clusters(COLOURS[subprogram[2]],int(subprogram[3]),None)
        elif "arg" in subprogram[1]:
            return find_clusters(COLOURS[subprogram[2]],int(subprogram[3]),subscenes[subprogram[1]])
        
        

prg = break_program_to_subprograms("clusterof all green 1 arg1 clusterof yellow 1 ans")
print(prg)                
ans = execute_subprogram(prg[0],None)
print(labels)
print(ans)
