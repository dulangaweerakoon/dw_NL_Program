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
#from keras.layers import GRU, Input
#from keras.layers import Dense
from tensorflow.python.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed
from tensorflow.python.keras.models import Model
from keras.layers import Embedding
#from keras.layers import RepeatVector
#from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
import os 
import numpy as np
#from attention_decoder import AttentionDecoder
from attention_keras.layers.attention import AttentionLayer
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

def define_attention_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	encoder_inputs = Input(shape=(src_timesteps, src_vocab), name='encoder_inputs')
	decoder_inputs = Input(shape=(tar_timesteps - 1, tar_vocab), name='decoder_inputs') 
	encoder_gru = GRU(n_units, return_sequences=True,return_state=True, name='encoder_gru')   
	encoder_out, encoder_state = encoder_gru(encoder_inputs)
	decoder_gru = GRU(n_units, return_sequences=True,return_state=True, name='decoder_gru')
	decoder_out, decoder_state = decoder_gru(decoder_inputs,initial_state=encoder_state)
	attn_layer = AttentionLayer(name='attention_layer')
	attn_out, attn_states = attn_layer([encoder_out, decoder_out])
	decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])
	dense = Dense(tar_vocab, activation='softmax', name='softmax_layer')
	dense_time = TimeDistributed(dense, name='time_distributed_layer')
	decoder_pred = dense_time(decoder_concat_input)
	#model = Model(inputs=[encoder_inputs, decoder_inputs],outputs=decoder_pred)
	model = Model(inputs=encoder_inputs,outputs=decoder_pred)
	return model
# load datasets
dataset = load_clean_sentences('cmd-prg-both.pkl')
train = load_clean_sentences('cmd-prg-train.pkl')
test = load_clean_sentences('cmd-prg-test.pkl')
print(dataset[1,0])
print(dataset[1,1])

cmd_tokenizer = create_tokenizer(dataset[:, 0])
cmd_vocab_size = len(cmd_tokenizer.word_index) + 1
cmd_length = max_length(dataset[:, 0])
print('Command Vocabulary Size: %d' % cmd_vocab_size)
print('Command Max Length: %d' % (cmd_length))

prg_tokenizer = create_tokenizer(dataset[:, 1])
prg_vocab_size = len(prg_tokenizer.word_index) + 1
prg_length = max_length(dataset[:, 1])
print('Program Vocabulary Size: %d' % prg_vocab_size)
print('Program Max Length: %d' % (prg_length))

trainX = encode_sequences(cmd_tokenizer, cmd_length, train[:, 0])
trainY = encode_sequences(prg_tokenizer, prg_length, train[:, 1])
trainY = encode_output(trainY, prg_vocab_size)


testX = encode_sequences(cmd_tokenizer, cmd_length, test[:, 0])
testY = encode_sequences(prg_tokenizer, prg_length, test[:, 1])
testY = encode_output(testY, prg_vocab_size)

#size = dataX.shape[0]
#trainX = []
#trainY = []
#testX = []
#testY = []
#for i in range(0,size):
#    seed = np.random.randint(0,10)
#    if seed < 9:
#        trainX.append(dataX[i])
#        trainY.append(dataY[i])
#    else:
#        testX.append(dataX[i])
#        testY.append(dataY[i])
#trainX = array(trainX)
#trainY = array(trainY)
#
#testX  = array(testX)
#testY  = array(testY)


# define model
#model2 = define_model(cmd_vocab_size, prg_vocab_size, cmd_length, prg_length, 24)
model = define_attention_model(cmd_vocab_size, prg_vocab_size, cmd_length, prg_length, 24)
model.compile(optimizer='adam', loss='categorical_crossentropy')
#model.load_weights("model.h5")
# summarize defined model
print(model.summary())
#print(model2.summary())
#plot_model(model, to_file='model.png', show_shapes=True)
# fit model
filename = 'attention_model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=5000, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=1)
