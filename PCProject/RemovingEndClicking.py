from scipy.io import wavfile

sr, signal = wavfile.read('Fabric12Coat.wav', sr=16000)
print(sr)

signal = signal[:-16000]

wavfile.write('Fabric12CoatClipped3.wav', rate = sr, data = signal)
