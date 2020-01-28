finalScore = 0.6
import csv
finalScoreList = [finalScore]
with open(r"HumanTestViaGuessingGame.csv", 'a') as f:


##fields=['first','second','third']
##with open(r'name', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(finalScoreList)
