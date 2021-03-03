# Lab 1 - build bag of word 
In this lab we will talk about vectors.\
For finding "How close the vector to each order" we will use : `|v1 - v2|`

For recognize text we can declare a `Dictionary` which contain words.\
We can recognize the text similarity using this vector and how close they are too each other.\
Words in the `Dictionary` can have priority using weighting word ( coordinate ).

## Python - new thing
### Basic python
* `my_dist = [(euclidean_distance(test_row,item),i,get_label(item)) for i,item in enumerate(training_set)]` == build list using for **classic**
### ex1
* fileContent[i] == re.sub("[^a-zA-Z ]","", fileContent[i]) == %s/not [a-zA-z ]//g
* np.empty((m,n)) == matlab: zeros(m,n)

### ex2
* for i,item in enumerate(dictionary) == get index in for each
* dictionary = dictionaryContent.split() # list of words

### ex3
* frequency = np.empty((numFiles,len(dictionary)),dtype=np.int64) == np array
* frequency[i,:] == np.array move all columns


## Task
1.	Open the files lab1_ex1.py and lab1_ex2.py and read the code. 
2.	Open the file lab1_ex3.py and read the code. 
3.	Calculate the frequency vector for each text (the using texts algebra.txt, liter.txt, calculus.txt) 
and build the frequency matrix for all texts. 
4.	Calculate distance matrix for all texts and analyze the texts.
5.	Analyze the texts similarity using distance matrix. 

6.	For comparison add two existing texts (algebra2.txt and liter2.txt). 
Read, write and preprocess the texts. 
7.	Calculate the frequency matrix for all texts using the dictionary built from “dictionary.txt”
8.	Calculate the frequency matrix for all texts using the dictionary built from “general_ dict.txt”
9.	Calculate distance matrix for all texts and analyze the texts.



