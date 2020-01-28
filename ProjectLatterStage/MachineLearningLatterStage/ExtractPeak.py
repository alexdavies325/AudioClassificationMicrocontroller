from scipy.io import wavfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt


a = 'will'
b = 'mary'
c = a + b
print(c)

file = 'TestWrapper.wav'
sr, wav = wavfile.read(file)
print("signal rate = ", sr)


##subprocess.call(['aplay', file])


signal = wav
peakValue = np.amax(signal)
originalPeakValue = peakValue
numPeaks = 0
while peakValue > (0.5*originalPeakValue):
    plt.plot(signal)
    plt.show()
    print("shape = ", signal.shape)
    numPeaks += 1
    print("numPeaks = ", numPeaks)
    peakValueIndex = np.argmax(signal)
    print("peakIndex = ", peakValueIndex)
    minWin = peakValueIndex - int((sr/10)*0.3)
    maxWin = peakValueIndex + int((sr/10)*0.7)
    signal = np.append(signal[:minWin], signal[maxWin:])
    peakValue = np.amax(signal)
    print("peakValue = ", peakValue)

##for peak in Peaks:
##    if peakValue > minWin and peakValue < minMax:
##        NewPeak = False
##        break
##if NewPeak == False:
##    continue
##else:
##    Peaks[numPeaks] = (minWin, maxWin)


fileSplit = file.split('.')
filename = fileSplit[0]

print(filename + 'Clipped.wav')
wavfile.write(filename + 'Clipped.wav', rate = sr, data = signal)
##
##subprocess.call(['aplay', filename + 'Clipped.wav'])


##import numpy as np
##Peaks = []
##print(Peaks)
##numPeaks = 1
##minWin = 1000
##maxWin = 2600
##RangeList = [minWin, maxWin]
##print(RangeList)
##Peaks.append([minWin, maxWin])
##print(Peaks)
##print(Peaks[0][0])
##for elem in Peaks:
##    print("elem = ", elem)
##NewPeak = 900
##NewNewPeak = True
##for peak in Peaks:
##    if NewPeak > peak[0] and NewPeak < peak[1]:
##        NewNewPeak = False
