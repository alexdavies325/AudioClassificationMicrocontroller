#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:08:00 2019

@author: alex
"""

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
from keras.preprocessing import image
import subprocess
##import MFCC

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

def plot_signals(signals):
    if NumSubplotRows == 1:
        fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=False,
                                 sharey=True, figsize=(20,5))
        fig.suptitle('Time Series', size=16)
        i = 0
        for y in range(NCOLS):
            axes[y].set_title(list(signals.keys())[i])
            axes[y].plot(list(signals.values())[i])
            axes[y].get_xaxis().set_visible(False)
            axes[y].get_yaxis().set_visible(False)
            i += 1
            
    elif NumSubplotRows == 2:
        fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=False,
                                 sharey=True, figsize=(20,5))
        fig.suptitle('Time Series', size=16)
        i = 0
        for x in range(NROWS):
            for y in range(NCOLS):
                print("i = ", i)
                if i == 8:
                    break
                else:
                    axes[x,y].set_title(list(signals.keys())[i])
                    axes[x,y].plot(list(signals.values())[i])
                    axes[x,y].get_xaxis().set_visible(False)
                    axes[x,y].get_yaxis().set_visible(False)
                    i += 1
            

def plot_fft(fft):
    if NumSubplotRows == 1:
        fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=True,
                                 sharey=True, figsize=(20,5))
        fig.suptitle('Fourier Transforms', size=16)
        i = 0
        for y in range(NCOLS):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[y].set_title(list(fft.keys())[i])
            axes[y].plot(freq, Y)
            axes[y].get_xaxis().set_visible(True)
            axes[y].get_yaxis().set_visible(True)
            i += 1
    
    elif NumSubplotRows == 2:
        fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=False,
                                 sharey=True, figsize=(20,5))
        fig.suptitle('Fourier Transforms', size=16)
        i = 0
        for x in range(NROWS):
            for y in range(NCOLS):
                print("i = ", i)
                if i == 8:
                    break
                else:
                    data = list(fft.values())[i]
                    Y, freq = data[0], data[1]
                    axes[x,y].set_title(list(fft.keys())[i])
                    axes[x,y].plot(freq, Y)
                    axes[x,y].get_xaxis().set_visible(False)
                    axes[x,y].get_yaxis().set_visible(False)
                    i += 1        
        

def plot_fbank(fbank):
    if NumSubplotRows == 1:
        fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=False,
                                 sharey=True, figsize=(20,5))
        fig.suptitle('Filter Bank Spectrograms', size=16)
        i = 0
        for y in range(NCOLS):
            axes[y].set_title(list(fbank.keys())[i])
            axes[y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[y].get_xaxis().set_visible(False)
            axes[y].get_yaxis().set_visible(False)
            i += 1
    
    elif NumSubplotRows == 2:
        fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=False,
                                 sharey=True, figsize=(20,5))
        fig.suptitle('Filter Bank Spectrograms', size=16)
        i = 0
        for x in range(NROWS):
            for y in range(NCOLS):
                print("i = ", i)
                if i == 8:
                    break
                else:
                    axes[x,y].set_title(list(fbank.keys())[i])
                    axes[x,y].imshow(list(fbank.values())[i],
                            cmap='hot', interpolation='nearest')
                    axes[x,y].get_xaxis().set_visible(False)
                    axes[x,y].get_yaxis().set_visible(False)
                    i += 1

def plot_mfccs(mfccs):
    if NumSubplotRows == 1:
        fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=False,
                                 sharey=True, figsize=(20,5))
        fig.suptitle('Mel Frequency Cepstrums', size=16)
        i = 0
        for y in range(NCOLS):
            axes[y].set_title(list(mfccs.keys())[i])
            axes[y].imshow(list(mfccs.values())[i],
                    cmap='inferno', interpolation='nearest')
            axes[y].get_xaxis().set_visible(False)
            axes[y].get_yaxis().set_visible(False)
            i += 1
    
    elif NumSubplotRows == 2:
        fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=False,
                                 sharey=True, figsize=(20,5))
        fig.suptitle('Mel Frequency Cepstrums', size=16)
        i = 0
        for x in range(NROWS):
            for y in range(NCOLS):
                print("i = ", i)
                if i == 8:
                    break
                else:
                    axes[x,y].set_title(list(mfccs.keys())[i])
                    axes[x,y].imshow(list(mfccs.values())[i],
                            cmap='inferno', interpolation='nearest')
                    axes[x,y].get_xaxis().set_visible(False)
                    axes[x,y].get_yaxis().set_visible(False)
                    i += 1

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y))/n
    return(Y, freq)


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

os.chdir("/home/alex/Alex/QMUL/Project2")


df = pd.read_csv('materials2.csv')
df.set_index('filename', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('Wavfiles/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate


classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)


print(classes)

signals = {}
signalsPitch = {}
signalsTime = {}
signalsNoise = {}
fft = {}
fbank = {}
mfccs = {}
mfccs2 = {}

fftPitch = {}
fftTime = {}
fftNoise = {}

fbankPitch = {}
fbankTime = {}
fbankNoise = {}

mfccsPitch = {}
mfccsTime = {}
mfccsNoise = {}

for c in classes:
    wav_file = df[df.label==c].iloc[5,0]
#    print("wav_file = ", wav_file)
    signal, rate = librosa.load('Wavfiles/'+wav_file, sr=16000)
#    rateWav, signalWav = wavfile.read('Wavefiles/'+wav_file)
#    print("signalWav = ", signalWav)
#    signal = signal[:16000]
#    mask = envelope(signal, rate, 0.0005)
#    signal = signal[mask]
#    print("signal = ", signal)
    maxValueIndex = np.argmax(signal)
    minWin = maxValueIndex - int((rate/10)*0.3)
    maxWin = maxValueIndex + int((rate/10)*0.7)
    signal = signal[minWin:maxWin]
    maxValue = max(signal)
    minValue = min(signal)
    signal = (signal - minValue) / (maxValue - minValue)
#    print(signal)
#    print("max of signal: ", max(signal))
    
    signalPitch = pitchShift(signal, rate, (np.random.randint(6) - 1))
#    print("signalPitch = ", signalPitch)
    signalTime = timeShift(signal, rate, 0.02, 'both')
    signalNoise = noiseInjection(signal, np.random.uniform(0, 0.05))
    
    signalsPitch[c] = signalPitch
    signalsTime[c] = signalTime
    signalsNoise[c] = signalNoise
    
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)
    fbank[c] = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    mfccResult = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    minValue = np.amin(mfccResult)
    maxValue = np.amax(mfccResult)
    mfccResult = (mfccResult - minValue)/(maxValue - minValue)
    mfccs[c] = mfccResult
    
    
    
    fftPitch[c] = calc_fft(signalPitch, rate)
    fbankPitch[c] = logfbank(signalPitch[:rate], rate, nfilt=26, nfft=1103).T
    mfccsPitch[c] = mfcc(signalPitch[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    
#    print("signalTime = ", signalTime)
    fftTime[c] = calc_fft(signalTime, rate)
    fbankTime[c] = logfbank(signalTime[:rate], rate, nfilt=26, nfft=1103).T
    mfccsTime[c] = mfcc(signalTime[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    
    fftNoise[c] = calc_fft(signalNoise, rate)
    fbankNoise[c] = logfbank(signalNoise[:rate], rate, nfilt=26, nfft=1103).T
    mfccsNoise[c] = mfcc(signalNoise[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
#    mfccs2[c] = MFCC.mfccNew(BigSample = signal[:rate], DESIRED_WINDOW_SIZE = 1103,
#         WINDOW_STEP = 441, ZeroPadWinSize = 2048, SampleRate = rate, n_mels = 26, n_coeff = 13).T/1000.0
#    print("mfcc.shape = ", mfccs[c].T.shape)
#    print("mfcc2.shape = ", mfccs[c].T.shape)
#    print("class = ", c)

##print(np.shape(mfccs['Glass']))
##print(np.shape(fbank['Glass']))

mfccsImage = image.img_to_array(mfccs['Glass'])

#print(os.getcwd())

image.save_img('image1.png', mfccsImage)
mfccsImage2 = image.load_img('image1.png', target_size=(299, 299))
image.save_img('image2.png', mfccsImage2)

    
NumSubplotRows = 2
NROWS = 2
NCOLS = 3
    
plot_signals(signals)
plt.show()
#
#plot_signals(signalsPitch)
#plt.show()
#
#plot_signals(signalsTime)
#plt.show()
#
#plot_signals(signalsNoise)
#plt.show()
#
#plot_fft(fft)
#plt.show()
#
#plot_fft(fftPitch)
#plt.show()
#
#plot_fft(fftTime)
#plt.show()
#
#plot_fft(fftNoise)
#plt.show()
#
plot_fbank(fbank)
plt.show()
#
#plot_fbank(fbankPitch)
#plt.show()
#
#plot_fbank(fbankTime)
#plt.show()
#
#plot_fbank(fbankNoise)
#plt.show()

plot_mfccs(mfccs)
plt.show()

#plot_mfccs(mfccsPitch)
#plt.show()

#plot_mfccs(mfccsTime)
#plt.show()
#
#plot_mfccs(mfccsNoise)
#plt.show()

#plot_mfccs(mfccs2)
#plt.show()

#print(mfccs['Glass'].T)
#print(mfccs2['Glass'].T)

if len(os.listdir("clean")) == 0:
    for f in tqdm(df.filename):
        signal, rate = librosa.load('Wavfiles/'+f, sr=16000)
#        print(signal.shape)
        # Remove first and last 20th of a second,
        # to remove clicking when starting/stopping recording.
        signal = signal[3200:-3200]
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='clean/'+f, rate=rate, data=signal[mask])
























