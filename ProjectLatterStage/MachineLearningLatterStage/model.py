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
import librosa
import time
#import MFCC

def pitchShift(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def noiseInjection(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def timeShift(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
            
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean() # if center != True, then it sets the value at the front of the window
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

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
    filePeaksPicked = {}
    for f in df.index:
        filePeaksPicked[f] = []
    x = []
    y = []
    _min, _max = float('inf'), -float('inf')
    totalNumPeaks = 0
    nextFile = False
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('clean/'+file)
        
        """ Count number of peaks in file """
        print("file = ", file)
        signal = wav
        peaks = []
        peakIndices = []
        numPeaks = 0
        peakValue = np.amax(signal)
        originalPeakValue = peakValue
        while peakValue > 0.5*originalPeakValue:
            peakValueIndex = np.argmax(signal)
            minWin = peakValueIndex - int((rate/10)*0.3)
            maxWin = peakValueIndex + int((rate/10)*0.7)
            signal = np.append(signal[:minWin], signal[maxWin:])
            if signal.shape[0] < int(rate/10):
                break
            if maxWin > (wav.shape[0]-(int((rate/10)*0.8))):
                print("Peak too close to end of file")
            elif minWin < 0:
                print("Peak too close to start of file")
            else:
                peaks.append(minWin)
                numPeaks += 1
                peakIndices.append(peakValueIndex)
                peakValue = np.amax(signal)
        print("totalNumPeaks = ", totalNumPeaks)
        totalNumPeaks += numPeaks
        print("numPeaks = ", numPeaks)
        print("totalNumPeaks = ", totalNumPeaks)
        
        """ Choose peak not already chosen """
        while True:
            if numPeaks == len(filePeaksPicked[file]):
                nextFile = True
                break
            peakChoiceIndex = np.random.randint(numPeaks)
            peakChoice = peakIndices[peakChoiceIndex]
            if peakChoice in filePeaksPicked[file]:
                continue
            else:
                filePeaksPicked[file].append(peakChoice)
                break
            
        if nextFile == True:
            nextFile = False
            _ -= 1
            continue
        
        """ Sample around chosen peak """
        print(peakChoice)
        startIndex = peaks[peakChoiceIndex]
        sample = wav[startIndex:(startIndex + config.step)]
        
        
        
#        rand_index = np.random.randint(0, wav.shape[0] - config.step)
#        sample = wav[rand_index:rand_index + config.step]
        ## shift at most 3 half steps left or right
        
        """ Add Data Augmentation """
#        sample = pitchShift(sample, rate, (np.random.randint(4) - 1))
#        sample = timeShift(sample, rate, 0.02, 'both')
#        sample = noiseInjection(sample, np.random.uniform(0, 0.02))
#        mask = envelope(sample, rate, 0.0005)
        
        """ Normalize before taking mfcc """
        maxValue = np.amax(sample)
        minValue = np.amin(sample)
        sample = (sample - minValue) / (maxValue - minValue)
        print("sample = ", sample)
        x_sample = mfcc(sample, rate,
                        numcep=config.nfeat, nfilt=config.nfilt,
                        nfft=config.nfft)
        
        """ Normalize after taking mfcc """
        minValueMfcc = np.amin(x_sample)
        maxValueMfcc = np.amax(x_sample)
        x_sample = (x_sample - minValueMfcc)/ (maxValueMfcc - minValueMfcc)
        print("sample = ", x_sample)
        _min = min(np.amin(x_sample), _min)
        _max = max(np.amax(x_sample), _max)
        x.append(x_sample)
        print("x_sample = ", x_sample)
        y.append(classes.index(rand_class))
        
    avgNumPeaks = totalNumPeaks / n_samples
    print("totalNumberPeaks = ", totalNumPeaks)
    print("Average number of peaks = ", avgNumPeaks)
    
    """ Normalize, taking min and max across all files """
#    config.min = _min
#    config.max = _max
    x, y = np.array(x), np.array(y)
#    print("x shape = ", x.shape)
#    x = (x - _min) / (_max - _min)
    
    """ Convert for entry to NN """
    if config.mode == 'conv':
        print("x.shape[0] = ", x.shape[0])
        print("x.shape[1] = ", x.shape[1])
        print("x.shape[2] = ", x.shape[2])
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    elif config.mode == 'time':
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    y = to_categorical(y, num_classes=3)
    
    """ Save configuation """
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
    model.add(Dense(3, activation='softmax'))
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
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

df = pd.read_csv('materials2.csv')
df.set_index('filename', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate
    
classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

#n_samples = int(1 * int(df['length'].sum()/0.1)) # number of samples dividing samples into 10th of seconds
n_samples = 3 * 140
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)


fig, ax = plt.subplots()
ax.set_title('Class Distribution After Clean', y=1.08)
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

startTrainTime = time.time()

model.fit(x, y, epochs=100, batch_size=32,
          shuffle=True,
          class_weight=class_weight,
          validation_split=0.4,
          callbacks=[checkpoint])

finishTrainTime = time.time()
totalTrainTime = finishTrainTime - startTrainTime
print(totalTrainTime)

model.save(config.model_path)

model.save('Data/model_' + config.mode + 'ReducedLessTest.h5')
print("Saved model to disk")
print(config.min)
print(config.max)






