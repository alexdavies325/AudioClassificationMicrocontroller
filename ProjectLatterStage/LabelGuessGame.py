# MaterialGuessGame

import os, random, subprocess, time

# Number of files in directory
directory = '/home/alex/Alex/QMUL/Project2/Machine learning2/Wavfiles'
listOfMaterials = ['wood', 'paper', 'plastic', 'metal', 'fabric', 'glass']
listNumbers = []

# list of ints to seperate label from filename
for num in range(10):
    listNumbers.append(str(num))
#print(listNumbers)

##cwd = os.getcwd()
##print(cwd)
##os.chdir(directory)

numberOfFiles = len([item for item in os.listdir(directory)
                       if os.path.isfile(os.path.join(directory, item))])

# Take roughly 10% of total number of files
Files10Perc = 10

def getRandChoice(directory):
    choice = random.choice(os.listdir(directory))
##    print(choice)
    # Get label/class from filename
    for char in choice:
        if char in listNumbers:
            splitOn = char
            break
    stringSplit = choice.split(splitOn, 1)
    stringClass = stringSplit[0].lower()
##    print(stringClass)
    return stringClass, choice

waitTime = 0.01

def slowPrint(msg, linebreak = True):
    for letter in msg:
        print(letter, end='')
        time.sleep(waitTime)
    if (linebreak):
        print('\n', end='')


# Ready
msg = "Are you ready? (y for yes, n for no) \r\n"
for letter in msg:
    print(letter, end='')
    time.sleep(waitTime)
    
firstAnswer = input(" ")
firstAnswer.lower()
time.sleep(1)
if firstAnswer == 'y' or firstAnswer == 'yes':
    msg = "Great.\n"
    for letter in msg:
        print(letter, end='')
        time.sleep(waitTime)
else:
    msg = "..I'll take that as a yes\n"
    for letter in msg:
        print(letter, end='')
        time.sleep(waitTime)
    
time.sleep(1)
msg = "Let's go!\n"
for letter in msg:
    print(letter, end='')
    time.sleep(waitTime)
    
time.sleep(1)
msg = "I'll play the sound you guess the material.\n"
for letter in msg:
    print(letter, end='')
    time.sleep(waitTime)
time.sleep(1)

slowPrint("Available answers are: ")
for option in listOfMaterials:
    slowPrint(option, linebreak = False)
    print(" ", end='')

print("")

# Change to file folder
os.chdir(directory)


# Keep track of scores
scores = []


slowPrint("Number of files = ", linebreak = False)
slowPrint(str(numberOfFiles))

slowPrint("Number of materials to guess: ", linebreak = False)
slowPrint(str(Files10Perc))

answer = ''
materialNumber = 0

# Check if sound was heard
# Collect Guess
while True:
    if answer != 'r':
        materialNumber += 1
        slowPrint("Material Number: ", linebreak=False)
        slowPrint(str(materialNumber))
        # Get random file from folder
        stringClass, choice = getRandChoice(os.getcwd())
        
        msg = "I have a sound..\n"
        for letter in msg:
            print(letter, end='')
            time.sleep(waitTime)
        time.sleep(1)
        
    msg = "Playing..\n"
    for letter in msg:
        print(letter, end='')
        time.sleep(waitTime)
    time.sleep(1)
    subprocess.call(['aplay', '-q', choice])
    msg = "Played\n"
    for letter in msg:
        print(letter, end='')
        time.sleep(waitTime)
    time.sleep(1)
    msg = "Your guess, hombre (press r to repeat sound): \n"
    for letter in msg:
        print(letter, end='')
        time.sleep(waitTime)
    answer = input("")
    # lowercase
    answer = answer.lower()
    if answer == 'r':
        continue
    elif answer in listOfMaterials:
        if answer == stringClass:
            slowPrint("Well done, holmes.")
            # Reveal actual file
            slowPrint("Correct label was: ", stringClass)
            slowPrint("filename: ", linebreak = False)
            slowPrint(choice)
            scores.append(1)
            if len(scores) == Files10Perc:
                break
        elif answer != stringClass:
            slowPrint("Sorry, boyo, wrong answer!")
            slowPrint("Correct label was: ", linebreak = False)
            slowPrint(stringClass)
            slowPrint("filename: ", linebreak = False)
            slowPrint(choice)
            scores.append(0)
            if len(scores) == Files10Perc:
                break
        else:
            slowPrint("I broke down, sugar, fix the code.")
    else:
        slowPrint("Answer not available, chaval, try again")
        slowPrint("Available answers are: ")
        for option in listOfMaterials:
            slowPrint(option)
            print("", end='')
        print(" ")
        answer = 'r'

# calculate final score percentage
finalScore = 0
for score in scores:
    finalScore += score

finalScore /= Files10Perc

# Print score/percent
print("Final score: ", finalScore)

# Write to csv
print("Adding score to csv..")
csvPath = "/home/alex/Alex/QMUL/Project2"
os.chdir(csvPath)
import csv
finalScoreList = [finalScore]
with open(r"HumanTestViaGuessingGame.csv", 'a') as f:
    writer = csv.writer(f)
    writer.writerow(finalScoreList)
print("Addition complete")
    
    
