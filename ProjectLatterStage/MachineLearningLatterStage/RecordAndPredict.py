#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:32:45 2019

@author: alex
"""

import subprocess
from scipy.io import wavfile
import os
import numpy as np
import time
import pickle
from python_speech_features import mfcc, logfbank
from keras.models import load_model
#import pandas as pd
import librosa
import matplotlib.pyplot as plt


def plot_signals(signals):
    if NumSubplotRows == 1:
        fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=False,
                                 sharey=True, figsize=(20,5))
        fig.suptitle('Time Series', size=16)
        i = 0
        for y in range(NCOLS):
            print("i = ", i)
            print("signals.keys(): ", signals.keys())
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
        fig.suptitle('Filter Bank Coefficients', size=16)
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
        fig.suptitle('Filter Bank Coefficients', size=16)
        i = 0
        for x in range(NROWS):
            for y in range(NCOLS):
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
        fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
        i = 0
        for y in range(NCOLS):
            axes[y].set_title(list(mfccs.keys())[i])
            axes[y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[y].get_xaxis().set_visible(False)
            axes[y].get_yaxis().set_visible(False)
            i += 1
    
    elif NumSubplotRows == 2:
        fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=False,
                                 sharey=True, figsize=(20,5))
        fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
        i = 0
        for x in range(NROWS):
            for y in range(NCOLS):
                axes[x,y].set_title(list(mfccs.keys())[i])
                axes[x,y].imshow(list(mfccs.values())[i],
                        cmap='hot', interpolation='nearest')
                axes[x,y].get_xaxis().set_visible(False)
                axes[x,y].get_yaxis().set_visible(False)
                i += 1



def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y))/n
    return(Y, freq)


#files = ['TestWrapper.wav', 'test-mic.wav', 'TestKitcheRoll.wav',
#         'TestPaper.wav', 'TestPlastic.wav', 'TestWrapper2.wav',
#         'TestWood.wav', 'Plastic1.wav', 'Paper10Serviette.wav', 'Glass7.wav',
#         'TestSnowPlastic.wav', 'TestGlass3.wav']

files = []

#for file in os.listdir('WavfilesReduced2Test'):
for file in os.listdir('TestWav'):
    files.append(file)
    
print(files)
    
    

#""" Record wavfile & play"""
#time.sleep(1)
#print("Recording..")
#MyOut = subprocess.Popen(['arecord', '-c', '1',
#                          '-d', '2', '-f', 'S16_LE',
#                          '-r', '16000', file], stdout=subprocess.PIPE,
#                         stderr=subprocess.STDOUT)
#stdout, stderr = MyOut.communicate()
#stdout = stdout.decode().rstrip()
#print("stdout = ", stdout)
#print("stderr = ", stderr)
#print("Recorded")
#print("Playing..")
#subprocess.call(['aplay', file])
#print("Played")

# Load configuartion
p_path = os.path.join('pickles', 'conv.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

# Load to model
model = load_model(config.model_path)
print("model loaded from: ", config.model_path)



signals = {}
fft = {}
fbank = {}
mfccs = {}

signalsNormalized = {}
fftNormalized = {}
fbankNormalized = {}
mfccsNormalized = {}

totalNumPeaks = 0

#os.chdir('WavfilesReduced2Test')
os.chdir('TestWav')

for file in files:
    signal, sr = librosa.load(file, sr=16000)
    
    wavfile.write(file, rate=sr, data=signal)
    # Load wavfile
    rate, signal = wavfile.read(file)
    
#    for i in range(10):
    #    signalPart = signal[(i*1600):((i*1600)+1600)]
#        signalPart = signal[:16000]
#        
        
#    #    #print(x.shape)
#    maxValue = np.amax(signal)
#    minValue = np.amin(signal)
##    x = (x - config.min)/ (config.max - config.min)
###    x = ((x - minValue)/ (maxValue - minValue))
##    print("Max value: ", np.amax(x))
#    signalNormalized = (signal - minValue)/(maxValue - minValue)
    
    signals[file] = signal
    fft[file] = calc_fft(signal, rate)
    fbank[file] = logfbank(signal[:rate], rate, nfilt=config.nfilt, nfft=config.nfft).T
    mfccs[file] = mfcc(signal[:rate], rate, numcep=config.nfeat,
                         nfilt=config.nfilt, nfft=config.nfft).T
         
    
#    signalsNormalized[file] = signalNormalized
#    fftNormalized[file] = calc_fft(signalNormalized, rate)
#    fbankNormalized[file] = logfbank(signalNormalized[:rate], rate, nfilt=config.nfilt, nfft=config.nfft).T
#    mfccsNormalized[file] = mfcc(signalNormalized[:rate], rate, numcep=config.nfeat,
#                         nfilt=config.nfilt, nfft=config.nfft).T
        
    
        ## Get random sample from glass and plastic classes
        #            if (RESULTSOBTAINED != 1):
        #                if GLASSRESULTOBTAINED == 0 and label == 'Glass' and counter > 100:
        #                    print("Assigning glass sample.")
        #                    print("glasslabel: ", label)
        #                    SampleGlass = x
        #                    GLASSRESULTOBTAINED = 1
        #                elif PLASTICRESULTOBTAINED == 0 and label == 'Plastic' and counter > 200:
        #                    print("Assigning plastic sample.")
        #                    print("plasticlabel: ", label)
        #                    SamplePlastic = x
        #                    PLASTICRESULTOBTAINED = 1
        #                elif GLASSRESULTOBTAINED == 1 and PLASTICRESULTOBTAINED == 1:
        #                    RESULTSOBTAINED = 1
        #                counter += 1
    print("file = ", file)
    
    originalSignal = signal
    tmpSignal = originalSignal
    peaks = []
    numPeaks = 0
    peakValue = np.amax(tmpSignal)
    originalPeakValue = peakValue
    while peakValue > 0.5*originalPeakValue:
        peakValueIndex = np.argmax(tmpSignal)
        minWin = peakValueIndex - int((rate/10)*0.3)
        maxWin = peakValueIndex + int((rate/10)*0.7)
        tmpSignal = np.append(tmpSignal[:minWin], tmpSignal[maxWin:])
        if tmpSignal.shape[0] < int(rate/10):
            break
        if maxWin > (originalSignal.shape[0]-(int((rate/10)*0.8))):
            print("Peak too close to end of file")
        elif minWin < 0:
            print("Peak too close to start of file")
        else:
            peaks.append(minWin)
            numPeaks += 1
            peakValue = np.amax(tmpSignal)
    totalNumPeaks += numPeaks
    print("numPeaks = ", numPeaks)
#    peakChoice = np.random.randint(len(peaks))
#    startIndex = peaks[peakChoice]
#    sample = originalSignal[startIndex:(startIndex + config.step)]
    
    
    for startIndex in peaks:
        print("startIndex = ", startIndex)
        sample = originalSignal[startIndex:(startIndex + config.step)]
        print("sample shape = ", sample.shape)
        minValue = np.amin(sample)
        maxValue = np.amax(sample)
        sample = (sample - minValue)/ (maxValue - minValue)
        
        x = mfcc(sample, rate, numcep=config.nfeat,
                         nfilt=config.nfilt, nfft=config.nfft)
        
        minValueMfcc = np.amin(x)
        maxValueMfcc = np.amax(x)
        x = (x - minValueMfcc)/ (maxValueMfcc - minValueMfcc)
        
        
        
        if config.mode == 'conv':
            x = x.reshape(1, x.shape[0], x.shape[1], 1)
        elif config.mode == 'time':
            x = np.expand_dims(x, axis=0)
        probabilities = model.predict(x)
        print(probabilities)
        
        prediction = np.argmax(probabilities)
        print(prediction)
        
#for file in files:
#    
#    print("file = ", file)
#    for i in range(10):
#        signalPart = signal[(i*1600):((i*1600)+1600)]
#        
#        x = mfcc(signal[:rate], rate, numcep=config.nfeat,
#                         nfilt=config.nfilt, nfft=config.nfft)
#        x = (x - config.min)/ (config.max - config.min)
#        
#        
#        
#        if config.mode == 'conv':
#            x = x.reshape(1, x.shape[0], x.shape[1], 1)
#        elif config.mode == 'time':
#            x = np.expand_dims(x, axis=0)
#        probabilities = model.predict(x)
#        print(probabilities)
#        
#        prediction = np.argmax(probabilities)
#        print(prediction)


NumSubplotRows = 3
NROWS = 3
NCOLS = 3

#print(signals)

plot_signals(signals)
plt.show()

plot_signals(signalsNormalized)
plt.show()

plot_fft(fft)
plt.show()

plot_fft(fftNormalized)
plt.show()

plot_fbank(fbank)
plt.show()

plot_fbank(fbankNormalized)
plt.show()

plot_mfccs(mfccs)
plt.show()

plot_mfccs(mfccsNormalized)
plt.show()
























