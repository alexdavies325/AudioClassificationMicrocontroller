import os
import pandas as pd

path = '/home/alex/Alex/QMUL/Project2/Machine learning2/WavfilesReduced2NoWoodOrMetal'

os.chdir(path)

## Csv columns
dict1= {'filename' : [], 'class' : []}


## List of single digit ints as strings, to check for location
## to split filename
listNumbers = []
for i in range(10):
    listNumbers.append(i)
print(listNumbers)
listNumbers = [str(x) for x in listNumbers]



for filename in os.listdir(path):
    
    print("filename: ", filename)
    
    ## Split location, to extract class from filename
    for char in filename:
        if char in listNumbers:
            splitOn = char
            break
        
    stringSplit = filename.split(char, 1)
    stringClass = stringSplit[0]
    print(stringClass)
    dict1['filename'].append(filename)
    dict1['class'].append(stringClass)

print(dict1)
df = pd.DataFrame(dict1)
print(df)
df.to_csv('materialsReduced.csv', index = False)




#### Test
##string = "Glass25.wav"
##print(string)
##
##for char in string:
##    print(char)
##    if char in listNumbers:
##        splitOn = char
##        break
##    
##print(splitOn)
##
##stringSplit = string.split(char)
##stringClass = stringSplit[0]
##print("stringClass = ", stringClass)
##dict1['filename'].append(string)
##dict1['class'].append(stringClass)
##print(dict1)
