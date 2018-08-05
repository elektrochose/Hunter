# Load Larger LSTM network and generate text
import re
import os
import sys
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
ROOT = '/home/pablo/python/'

if not os.path.isfile(ROOT + 'Hunter/input_XY.p'):
	raw_text = open(ROOT + 'Hunter/input.txt', 'r').read()
	raw_text = raw_text.lower()
	raw_text = re.sub(r'[^\x00-\x7f]',r'', raw_text)

	# create mapping of unique chars to integers, and a reverse mapping
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))
	# summarize the loaded data
	n_chars = len(raw_text)
	n_vocab = len(chars)
	print "Total Characters: ", n_chars
	print "Total Vocab: ", n_vocab
	# prepare the dataset of input to output pairs encoded as integers
	seq_length = 100
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	print "Total Patterns: ", n_patterns
	# reshape X to be [samples, time steps, features]
	X = np.reshape(dataX, (n_patterns, seq_length, 1))
	# normalize
	X = X / float(n_vocab)
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)
	pickle.dump([X,y], open(ROOT + 'Hunter/input_XY.p','wb'))
else:
	print 'loading data...'
	X,y = pickle.load(open(ROOT + 'Hunter/input_XY.p', 'rb'))


'''
TO-DO LIST:
Split into training and validation sets properly
Set up new folder for new models
Track training and validation loss/accuracy over epochs

'''
initial_epoch = 0
no_units = 512
model_dir = 'Hunter/models/'
# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(512))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))


#load best model if it exists and start from there
if os.listdir(ROOT + model_dir):
    models_so_far = os.listdir(ROOT + model_dir)
    models_so_far.sort()
    best_model = models_so_far[-1]
    dashes = [w.start() for w in re.finditer(r"-", best_model)]
    initial_epoch = int(best_model[dashes[1] + 1: dashes[2]])
    print 'best model so far: %s, epoch: %i' %(best_model, initial_epoch)
    model.load_weights(ROOT + model_dir + best_model)

model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = ROOT + model_dir + 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath,
 							 monitor='loss',
							 verbose=1,
							 save_best_only=True,
							 mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, epochs=100, batch_size=512,
 				callbacks=callbacks_list,
				initial_epoch=initial_epoch)