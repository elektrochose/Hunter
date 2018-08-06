import os
import sys
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import load_model


ROOT = '/Users/pablomartin/python/'
HOME = 'Hunter/'
models_available = os.listdir(ROOT + HOME + 'models256/')
model_loss = [float(str((w[w.find('.')-1:-5]))) for w in models_available]
best_model = models_available[np.argmin(model_loss)]



# load the LSTM model
model_config = pickle.load(open(ROOT + 'Hunter/Extras/network_config256.p', 'rb'))
model = Sequential.from_config(model_config)
n_vocab = model.output.shape[1].value
print 'best model so far: %s' %(best_model)
model.load_weights(ROOT + 'Hunter/models256/' + best_model)


#load number to character mapping
int_to_char = pickle.load(open(ROOT + 'Hunter/Extras/int_to_char.p', 'rb'))

# pick a random seed
seeds = pickle.load(open(ROOT + 'Hunter/Extras/seeds.p', 'rb'))
start = np.random.randint(0, len(seeds)-1)
pattern = seeds[start]


print '\n' + '*' * 80
print 'GENERATED OUTPUT'
print '*' * 80 + '\n'
# generate characters
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)

    by_max = np.argmax(prediction)
    by_probability = int(np.digitize(np.random.random(), np.cumsum(prediction)))
    options = [by_max, by_probability]
    index = options[np.random.random() > 0.4]

    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print "\n\n" + '*' * 80
