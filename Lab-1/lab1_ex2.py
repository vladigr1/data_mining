# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:03:47 2020

@author: ראובן
"""

# import regular expressins packge
# import numbers package
import re
import numpy as np

numFiles = 3

fileContent = ["","",""] #list - like in python
#read first file
file = open('algebra.txt','r')
for line in file:
    fileContent[0] += line
    
#read second file
file = open('liter.txt','r')
for line in file:
    fileContent[1] += line     
#read third file
file = open('calculus.txt','r')
for line in file:
    fileContent[2] += line 

# Remove extra spaces
# Remove non-letter chars    
# Change to lower case
for i in range(0,numFiles): # i = 0:numFiles-1
    fileContent[i] = re.sub(" +"," ", fileContent[i])
    fileContent[i] = re.sub("[^a-zA-Z ]","", fileContent[i])
    fileContent[i] = fileContent[i].lower()

# Read dictionary file
dictionaryFile = open('dictionary.txt','r')
dictionaryContent = ""
for line in dictionaryFile:
    dictionaryContent += line

# make dictionary list    
dictionary = dictionaryContent.split() # list of words

# count the number of dictionary words in files
frequency = np.empty((numFiles,len(dictionary))) # like  matlab : zeros(m,n) but with garbage
for i in range(0,numFiles):
    for j,word in enumerate(dictionary): # iterate index and word
        print(j,word,"\n")
        frequency[i,j] = len(re.findall(word,fileContent[i])) # count array 
        
# find the distance matrix between the text files
dist = np.empty((numFiles,numFiles))
for i in range(0,numFiles): 
    for j in range(0,numFiles):
        dist[i,j] = np.linalg.norm(frequency[i,:]-frequency[j,:]) # np.array is like matlab array [1,1:3]
        
print("dist=\n",dist)   
  
    
 
    