import os
import pandas as pd

path = '/home/alex/Alex/QMUL/Project2/Machine learning/Wavefiles'

os.chdir(path)

dict1= {'filename' : [], 'label' : []}

for filename in os.listdir(path):
    with open(filename) as f:
        stringSplit = filename.split("B")
        stringName = stringSplit[0]
        print(stringName)
        dict1['filename'].append(filename)
        dict1['label'].append(stringName)


##pathSplit = path.split("/")
##print(pathSplit)
##path = ''
##print(len(pathSplit)-1)
##
##for i in range(len(pathSplit)-1):
##    print(i)
##    path = path + '/' + pathSplit[i]
##
##
##for i in path:
##
##
for i in range(1, len(path)+1):
    if path[-i] == '/':
        path = path[:-i]
        break
        
os.chdir(path)
df = pd.DataFrame(dict1)
df.to_csv('materials.csv', index = False)
