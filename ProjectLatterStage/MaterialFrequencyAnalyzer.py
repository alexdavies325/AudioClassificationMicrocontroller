import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


path = "WavfilesReduced"

def calc_rfft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y))/n
    return(Y, freq)

# Make lists for each material
GlassFreqs = []
PlasticFreqs = []
PaperFreqs = []

materialFreqs = {}

ListNumAsStr = []
for num in range(10):
    ListNumAsStr.append(str(num))

# Get label from file
for file in os.listdir(path):
    print("file = ", file)
    sr, signal = wavfile.read(path + "/" + file)


    for char in file:
        if char in ListNumAsStr:
            splitOn = char
            break
        
    fileSplit = file.split(splitOn)
    label = fileSplit[0]
    print(label)

    # Perform rfft
    Y, freq = calc_rfft(signal, sr)

    # Get max frequency
    maxIndex = np.argmax(Y)
    maxFreq = freq[maxIndex]
    print("maxIndex = ", maxIndex)
    print("maxFreq = ", maxFreq)

##    plt.plot(freq, Y)
##    plt.show()

    # Append frequencies from each material
    if label == 'Glass':
        GlassFreqs.append(maxFreq)
    elif label == 'Plastic':
        PlasticFreqs.append(maxFreq)
    elif label == 'Paper':
        PaperFreqs.append(maxFreq)
    else:
        print("Label Error")

# Have a look at the frequency ranges

print("GlassFreqs = ", GlassFreqs)
print("max glass freq = ", max(GlassFreqs))
print("PlasticFreqs = ", PlasticFreqs)
print("max plastic freq = ", max(PlasticFreqs))
print("PaperFreqs = ", PaperFreqs)
print("max paper freq = ", max(PaperFreqs))
# Use the frequency ranges on test samples



















    
