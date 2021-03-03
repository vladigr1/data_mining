# -*- coding: utf-8 -*-

# import regular expressins packge
# import numbers package
import re
import numpy as np

def readFile(fileName):
    file = open(fileName,'r')
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

def vectorlizeFileStr(dictList, fileStr):
    # count the number of dictionary words in files
    vectorFile = np.empty((len(dictList)),dtype=np.int64)
    for i,word in enumerate(dictList):
        vectorFile[i] = len(re.findall(word,fileStr))
    return vectorFile


if __name__ == "__main__":
    fileNameList = ['algebra.txt', 'calculus.txt', 'liter.txt', 'algebra2.txt', 'liter2.txt' ]
    fileStrList = []
    for fileName in fileNameList:
        fileStrList.append(preProcess(readFile(fileName)))

    dictList = readFile('./dictionary.txt').split()
    frequencyMatrix = np.empty((len(fileNameList),len(dictList)),dtype=np.float64)
    for i in range(len(fileNameList)):
        frequencyMatrix[i][:] = vectorlizeFileStr(dictList, fileStrList[i])

    # find the distance matrix 
    dist = np.empty((len(fileNameList),len(fileNameList)) )
    for i in range(len(fileNameList)): 
        for j in range(len(fileNameList)):
            dist[i,j] = np.linalg.norm(frequencyMatrix[i,:]-frequencyMatrix[j,:])
    print("Row Freq:\n\n")
    print(dist)     

    # normalizatio
    for i in range(len(fileNameList)):
        frequencyMatrix[i][:] = frequencyMatrix[i][:] / np.sum(frequencyMatrix[i][:])

    # find the distance matrix 
    dist = np.empty((len(fileNameList),len(fileNameList)) )
    for i in range(len(fileNameList)): 
        for j in range(len(fileNameList)):
            dist[i,j] = np.linalg.norm(frequencyMatrix[i,:]-frequencyMatrix[j,:])
    print("\n\nNormlized Freq:\n\n")
    print(dist)     