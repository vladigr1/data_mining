# import regular expressins packge
# import numbers package
import numpy as np
import re

def readFile(fileName):
    file = open(fileName,'r',encoding="cp437")
    fileStr = ""
    for line in file:
        fileStr += line
    return fileStr
        
# Remove extra spaces
# Remove non-letter chars    
# Change to lower 
def preProcess(fileStr):
    fileStr = re.sub(" +"," ", fileStr)
    fileStr = re.sub("[^a-zA-Z ]","", fileStr)
    fileStr = fileStr.lower()
    return fileStr
            
rows = 6
fileContent = [""]*rows

#read  and preprocess files 
fileContent1 = preProcess(readFile('Eliot.txt'))
fileContent2 = preProcess(readFile('Tolkien.txt'))
# split 2 part
numParts = 2
partLength = int(len(fileContent1)/numParts) 
fileContent[0]  = fileContent1[0:partLength]
fileContent[1]  = fileContent1[partLength:]
numParts = 4
# split the third file to parts
partLength = int(len(fileContent2)/numParts) 
fileContent[2]  = fileContent2[0:partLength]
fileContent[3]  = fileContent2[partLength:partLength*2]
fileContent[4]  = fileContent2[partLength*2:partLength*3]
fileContent[5]  = fileContent2[partLength*3:]
#___________________________________ 
# construct DICTIONARY concat files contents
numFiles = rows
allFilesStr = ""
for i in range(numFiles):
    allFilesStr += fileContent[i]

# generate a set of all words in files 
wordsSet =  set(allFilesStr.split())

# Read stop words file - words that can be removed
stopWordsSet = set(readFile('stopwords_en.txt').split())
# Remove the stop words from the word list
dictionary = wordsSet.difference(stopWordsSet)
#_______________________________________
# count the number of dictionary words in files
wordFrequency = np.empty((rows,len(dictionary)),dtype=np.int64)
for i in range(rows):
    for j,word in enumerate(dictionary):
        wordFrequency[i,j] = len(re.findall(word,fileContent[i]))
        
# find the distance matrix between the text files
dist = np.empty((rows,rows))
for i in range(rows): 
    for j in range(rows):
        # calculate the distance between the frequency vectors
        dist[i,j] = np.linalg.norm(wordFrequency[i,:]-wordFrequency[j,:])
        
        
# Part 2:
# find the sum of the frequency colomns and select colomns having sum > 20
minSum = 20
sumArray =  wordFrequency.sum(axis=0)
indexArray = np.where(sumArray > minSum)

indexArraySize = len(indexArray[0])
wordFrequency1 = np.empty((rows,indexArraySize),dtype=np.int64)

# generate a freq matrix
for j in range(indexArraySize):
    wordFrequency1[:,j] = wordFrequency[:,indexArray[0][j]]

# generate the distance matrix
dist1 = np.empty((rows,rows))
for i in range(rows):
    for j in range(rows):
        dist1[i,j] = np.linalg.norm(wordFrequency1[i,:] - wordFrequency1[j,:])

print("dist=\n",dist)
print("dist=\n",dist1)
