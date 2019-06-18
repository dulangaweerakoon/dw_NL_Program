# Training Encoder-Decoder model to represent word embeddings and finally
# save the trained model as 'model.h5'

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="3,4,6"

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))


# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer


# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y


# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model


# load datasets
dataset = load_clean_sentences('cmd-prg.pkl')
train = load_clean_sentences('cmd-prg.pkl')
test = load_clean_sentences('cmd-prg.pkl')
print(dataset[1,0])
print(dataset[1,1])
# prepare english tokenizer
cmd_tokenizer = create_tokenizer(dataset[:, 0])
cmd_vocab_size = len(cmd_tokenizer.word_index) + 1
cmd_length = max_length(dataset[:, 0])
print('Command Vocabulary Size: %d' % cmd_vocab_size)
print('Command Max Length: %d' % (cmd_length))

# prepare german tokenizer
prg_tokenizer = create_tokenizer(dataset[:, 1])
prg_vocab_size = len(prg_tokenizer.word_index) + 1
prg_length = max_length(dataset[:, 1])
print('Program Vocabulary Size: %d' % prg_vocab_size)
print('Program Max Length: %d' % (prg_length))

dataX = encode_sequences(cmd_tokenizer, cmd_length, train[:, 0])
dataY = encode_sequences(prg_tokenizer, prg_length, train[:, 1])
dataY = encode_output(dataY, prg_vocab_size)

trainX = dataX[0:90]
trainY = dataY[0:90]

testX  = dataX[90:]
testY  = dataY[90:]
# define model
model = define_model(cmd_vocab_size, prg_vocab_size, cmd_length, prg_length, 32)
model.compile(optimizer='adam', loss='categorical_crossentropy')
#model.load_weights("model.h5")
# summarize defined model
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=5000, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=1)