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
from CustomCheckPoint import CustomModelCheckPoint

# load ascii text and covert to lowercase
ROOT = '/home/pablo/python/'

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

model.compile(loss = 'categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer = 'adam')

filepath = ROOT + model_dir + 'weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath,
 							 monitor='val_acc',
							 verbose=1,
							 save_best_only=True)
history_check = CustomModelCheckPoint(ROOT + model_dir + 'loss_acc.p')
callbacks_list = [checkpoint, history_check]

# fit the model
model.fit(X, y, epochs = 100,
 				batch_size = 512,
 				callbacks = callbacks_list,
                validation_split = 0.05,
				initial_epoch = initial_epoch)
