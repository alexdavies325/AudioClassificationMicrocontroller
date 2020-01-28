## Uses Bash function to convert mp3 files to wav format
import subprocess
import os

path = '/home/alex/Alex/QMUL/Project2/Machine learning2/Test'

os.chdir(path)

for filename in os.listdir(path):
    print("Original filename = ", filename)
    
    ## remove .mp3 from filename
    filenameNoType = filename.split('.', 1)
    print(filenameNoType)

    ## add .wav to filename
    newFilename = filenameNoType[0] + '.wav'
    print("New filename = ", newFilename)

    ## use bash function to convert mp3 to wav
    subprocess.call(['ffmpeg', '-i', filename, newFilename])


####Test
##stringTest = 'willy.mp3'
##filenameNoType = stringTest.split('.', 1)
##print(filenameNoType)
##
##stringTestComplete = filenameNoType[0] + '.wav'
##print(stringTestComplete)
