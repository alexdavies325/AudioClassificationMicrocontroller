import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy

""" Directory Check & Change """
##os.chdir('/home/alex/Alex')
##print("Directory: ", os.getcwd())

file = 'TestWrapper.wav'

##""" Record wavfile & play"""
##import subprocess
##MyOut = subprocess.Popen(['arecord', '-c', '1',
##                          '-d', '2', '-f', 'S16_LE',
##                          '-r', '44100', file], stdout=subprocess.PIPE,
##                         stderr=subprocess.STDOUT)
##stdout, stderr = MyOut.communicate()
##stdout = stdout.decode().rstrip()
##print("stdout = ", stdout)
##print("stderr = ", stderr)
##subprocess.call(['aplay', file])


""" Data from Wavefile """
sampleRate, data = wavfile.read(file)
print(data)
print(sampleRate)
print("data shape = ", data.shape)
count = 0
previousData = 0
sample = 0
for i in range(441):
    """sample += 1
    if data[i] != previousData:
        count += 1
        print("count: ", count)
        print("sample: ", sample)
        sample = 0
    print(data[i])
    previousData = data[i]"""

##""" X-axis of FFT/frequency plot """
##freqHz = []
##for i in range(len(data)):
##    freqHz.append(i)
##freqHz = [x*(sampleRate/len(data)) for x in freqHz]
##freqHz = freqHz[:int(len(freqHz)/2)+1]
##freqHz = numpy.array(freqHz)
##print(freqHz)
##print(len(freqHz))
print("data shape: ", data.shape[0])
freqHz = numpy.fft.rfftfreq(data.shape[0], sampleRate)
print("freqHz shape: ", freqHz.shape)

""" Perform FFT """
dataFFT = numpy.fft.rfft(data) # real fft cuts off repeated frequency information
print("dataFFT: ", dataFFT.shape)
print(dataFFT.shape)
dataFFFT = numpy.fft.fft(data)
print(dataFFFT)
print(dataFFFT.shape)

""" Plot """
#plt.plot(freqHz, abs(dataFFT))
plt.plot(data)
plt.show()
