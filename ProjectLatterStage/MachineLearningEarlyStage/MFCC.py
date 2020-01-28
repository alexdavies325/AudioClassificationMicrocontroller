#!/usr/bin/env python
# coding: utf-8

#   This software component is licensed by ST under BSD 3-Clause license,
#   the "License"; You may not use this file except in compliance with the
#   License. You may obtain a copy of the License at:
#                        https://opensource.org/licenses/BSD-3-Clause


"""KWS Feature Extraction example."""

import numpy as np
import librosa
import scipy
from scipy.signal import hann
from scipy.fftpack import dct

# take 400 samples from sample apart from last window
# zero pad to 512
def mfccNew(BigSample, DESIRED_WINDOW_SIZE, WINDOW_STEP, ZeroPadWinSize, SampleRate, n_mels, n_coeff):
    BigSampleLessFirstWin = BigSample[DESIRED_WINDOW_SIZE:]
    numWindows = len(BigSampleLessFirstWin)/WINDOW_STEP
    numWindowsInt = int(numWindows)
    numWindowsTrunc = float(numWindowsInt)
    if (numWindows - numWindowsTrunc) > 0:
        numWindows = numWindowsInt + 1
    numWindows += 1
    InitializeSpectrogram = [0] * n_coeff
    Spectrogram = np.array([InitializeSpectrogram])
    for i in range(numWindows):
        StartWindowIndex = i * WINDOW_STEP
        EndWindowIndex = StartWindowIndex + DESIRED_WINDOW_SIZE
        if EndWindowIndex > len(BigSample):
            EndWindowIndex = len(BigSample)
        RealWindowSize = EndWindowIndex - StartWindowIndex
        LittleSample = BigSample[StartWindowIndex:EndWindowIndex]
        print("LittleSample = ", LittleSample)
#        LittleSample = LittleSample/4095
        zeroPad = [0]* (ZeroPadWinSize - RealWindowSize)
        print(zeroPad)
        LittleSample = np.concatenate((LittleSample, zeroPad), axis=0)
        print("type = ", type(LittleSample))
        print(LittleSample.shape[0])
        Spectrogram = np.concatenate((Spectrogram, np.array([mfcc_col(LittleSample, LittleSample.shape[0], n_mels, n_coeff, SampleRate)])),
                                     axis=0)
    Spectrogram = Spectrogram[1:Spectrogram.shape[0]]
    print(Spectrogram.shape[0])
    return Spectrogram



def mfcc_col(buff_test, window, n_mels, n_coeff, SampleRate):

#    window = 512
    half_window = int(window / 2)
#    n_mels = 26
#    n_coeff = 13
    
    print(buff_test.shape)
    assert buff_test.shape == (window,)

    hann_asym_f32 = hann(window, sym=False).astype('float32')
    assert hann_asym_f32.shape == (window,), hann_asym_f32.shape

    buff_hann = buff_test * hann_asym_f32
    assert buff_hann.shape == (window,), buff_hann.shape

    fft = np.fft.fft(buff_hann, window)[:half_window + 1]
    assert fft.shape == (half_window + 1,), fft.shape

    ps = np.abs(fft)**2
    assert ps.shape == (half_window + 1,)

    mel = librosa.filters.mel(SampleRate, window, n_mels)
    assert mel.shape == (n_mels, half_window + 1)

    energy = np.dot(mel, ps)
    assert energy.shape == (n_mels,)

    logamplitude = 10 * np.log10(energy)
    assert logamplitude.shape == (n_mels,)

    dct_out = dct(logamplitude, type=3)
    assert dct_out.shape == (n_mels,)

    return(dct_out[1:(n_coeff + 1)])


# buffer_bus_01 is made of first 2048 samples of "bus.wav" file
#sr, ys = scipy.io.wavfile.read("bus.wav")
#
#buffer_01 = ys[0:2048]
#
#mfcc_col = mfcc_col(buffer_01)
#
#print('mfcc = ', mfcc_col[:])
    
BigSampleExt = [1]*1600
BigSampleExt = np.array(BigSampleExt)
print(BigSampleExt)
Spectrogram = mfccNew(BigSampleExt, 400, 160, 512, 16000, 26, 13)
print(Spectrogram)
print(Spectrogram.shape)
