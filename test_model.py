from pickle import load
from numpy import array, argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,7"
def load_dataset(filename):
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

# Map an integer to a word
def map_int_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Predict the target sequence
def predict_sequence(model, tokenizer, source):
    pred = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in pred]
    target = list()
    for i in integers:
        word = map_int_to_word(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

# Evaluate the model
def evaluate_model(model, tokenizer, source, raw_dataset):
    predicted, actual = list(), list()
    score = 0;
    length = 0;
    for i, source in enumerate(source):
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_source, raw_target = raw_dataset[i]
        if i < 100:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_source, raw_target, translation))
        if (raw_target == translation):
            score+=1    
        length+=1
    return (score/length)*100;
# Load datasets
dataset = load_dataset('cmd-prg-both.pkl')
train = load_dataset('cmd-prg-train.pkl')
test = load_dataset('cmd-prg-test.pkl')
#print(dataset[0,1])
prg_tokenizer = create_tokenizer(dataset[:, 1])
prg_vocab_size = len(prg_tokenizer.word_index) + 1
prg_length = max_length(dataset[:, 1])


cmd_tokenizer = create_tokenizer(dataset[:, 0])
cmd_vocab_size = len(cmd_tokenizer.word_index) + 1
cmd_length = max_length(dataset[:, 0])

# Prepare data
trainX = encode_sequences(cmd_tokenizer, cmd_length, train[:, 0])
testX = encode_sequences(cmd_tokenizer, cmd_length, test[:, 0])
#testX = trainX[90:]

#print(train[0,0])

model = load_model('model.h5')
#
#print('Testing on trained examples')
#accuracy = evaluate_model(model, prg_tokenizer, trainX, train)
#print("Training Accuracy: ",accuracy)
#print('Testing on test examples')
#accuracy = evaluate_model(model, prg_tokenizer, testX, test)
#print("Testing Accuracy: ",accuracy)

while 1:
    text = input("Type Command: ")
    text = encode_sequences(cmd_tokenizer, cmd_length,[text])
    print("Command: ",predict_sequence(model,prg_tokenizer,text))
