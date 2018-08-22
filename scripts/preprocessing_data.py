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

'''
TO-DO LIST:
Split into training and validation sets properly
Set up new folder for new models
Track training and validation loss/accuracy over epochs
'''


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
	#saving data bulk
	pickle.dump([X,y], open(ROOT + 'Hunter/Extras/input_XY.p','wb'))
	#sampling 100 random entries for seeds
	sub_dataX = [dataX[w] for w in np.random.randint(0,len(dataX),100)]
    pickle.dump(sub_dataX, open(ROOT + 'Hunter/Extras/seeds.p' , 'wb'))
    pickle.dump(int_to_char, open(ROOT + 'Hunter/Extras/int_to_char.p', 'wb'))
else:
	print 'loading data...'
	X,y = pickle.load(open(ROOT + 'Hunter/Extras/input_XY.p', 'rb'))
