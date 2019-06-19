import pickle as pkl
from numpy import random

def load_clean_data(filename):
    file = open(filename, 'rb')
    return pkl.load(file)

def save_clean_data(sentences, filename):
    pkl.dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


raw_data = load_clean_data('cmd-prg.pkl')
#print(raw_data.shape)
dataset = raw_data
random.shuffle(dataset)
#
train_set = dataset[:220, :]
test_set = dataset[220:, :]
#
save_clean_data(dataset, 'cmd-prg-both.pkl')
save_clean_data(train_set, 'cmd-prg-train.pkl')
save_clean_data(test_set, 'cmd-prg-test.pkl')
