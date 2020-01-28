#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 12:07:30 2019

@author: alex
"""
import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from keras.callbacks import ModelCheckpoint
from cfg import Config
import MFCC

def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None


def build_rand_feat():
#    tmp = check_data()
#    if tmp:
#        return tmp.data[0], tmp.data[1]
    x = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('clean/'+file)
        rand_index = np.random.randint(0, wav.shape[0] - config.step)
        sample = wav[rand_index:rand_index + config.step]
        x_sample = mfcc(sample, rate,
                        numcep=config.nfeat, nfilt=config.nfilt,
                        nfft=config.nfft)
#        x_sample = MFCC.mfccNew(BigSample = sample, DESIRED_WINDOW_SIZE = 400, \
#                              WINDOW_STEP = config.step, ZeroPadWinSize = config.nfft, \
#                              SampleRate = rate, n_mels = config.nfilt, n_coeff = config.nfeat)
        #mfccNew(BigSample, DESIRED_WINDOW_SIZE, WINDOW_STEP, ZeroPadWinSize, SampleRate, n_mels, n_coeff):
        _min = min(np.amin(x_sample), _min)
        _max = max(np.amax(x_sample), _max)
        x.append(x_sample)
        y.append(classes.index(rand_class))
    config.min = _min
    config.max = _max
    x, y = np.array(x), np.array(y)
    x = (x - _min) / (_max - _min)
    if config.mode == 'conv':
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    elif config.mode == 'time':
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    y = to_categorical(y, num_classes=2)
    
    config.data = (x, y)
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    
    return x, y


def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1,1),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

def get_recurrent_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True,
                     input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model
    


df = pd.read_csv('materials.csv')
df.set_index('filename', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate
    
classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2 * int(df['length'].sum()/0.1) # number of samples dividing samples into 10th of seconds
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)


fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()


config = Config(mode='conv')

x, y = build_rand_feat()
y_flat = np.argmax(y, axis=1)

if config.mode == 'conv':
    print('conv')
    input_shape = (x.shape[1], x.shape[2], 1)
    model = get_conv_model()
    
if config.mode == 'time':
    print('time')
    input_shape = (x.shape[1], x.shape[2])
    model = get_recurrent_model()
    
class_weight = compute_class_weight('balanced',
                                    np.unique(y_flat),
                                    y_flat)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc',
                             verbose=1, mode='max',
                             save_best_only=True,
                             save_weights_only=False,
                             period=1)


model.fit(x, y, epochs=10, batch_size=32,
          shuffle=True,
          class_weight=class_weight,
          validation_split=0.1,
          callbacks=[checkpoint])

model.save(config.model_path)

#model_json = model.to_json()
#with open("Data/model.json", "w") as json_file:
#    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))
#
#model.save_weights("Data/model.h5")
#print("Saved model to disk")


model.save('Data/model_' + config.mode + '2.h5')
print("Saved model to disk")
print(config.min)
print(config.max)






