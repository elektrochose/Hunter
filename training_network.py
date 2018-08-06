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
Track training and validation loss/accuracy over epochs
'''

#loading data
X, y = pickle.load(open(ROOT + 'Hunter/Extras/input_XY.p','rb'))



initial_epoch = 0
dropout = 0.5
no_units = 512
model_dir = 'Hunter/models512/'
# define the LSTM model
model = Sequential()
model.add(LSTM(no_units, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(no_units))
model.add(Dropout(dropout))
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
history = model.fit(X, y, epochs=100,
 					batch_size=512,
 					callbacks=callbacks_list,
					initial_epoch=initial_epoch)
pickle.dump(history.history, open(ROOT + model_dir + 'acc_loss_history.p' ,'wb'))
