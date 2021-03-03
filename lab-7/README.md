# Lab 7: Support Vector Machine - SVM
# python - new things
* panda - library for analyzing data\
  Dataframe - Two-dimensional, size-mutable, potentially heterogeneous tabular data. \
  Data structure also contains labeled axes (rows and columns). Arithmetic operations align on both row and column labels. Can be thought of as a dict-like container for Series objects. The primary pandas data structure.
* `confusion_table.values` == transfer values to numpy.array
* `irisdata = pd.read_csv(url, names=colnames)` == read csv and name columns
* `from sklearn.svm import SVC`\
  `svclassifier = SVC(kernel='poly', degree=8)` == svm classfier\
  `svclassifier.fit(data_train, real_label_train)` == train classifier\
  `pred_label_test = svclassifier.predict(data_test)` == test classifier on test set
# Tasks to do:
1.	Open the files Lab_7_0_SVM.py.  
2.	Load the data from Iris Database and choose 2 groups of the points.  
3.	Choose randomly n (n-parameter) points from each group (see species array) as a Training Set and rest of the points as a Testing Set.
4.	Construct SVM Model 
5.	Show the SVM partitioning results via different colors and signs. Show the original labelling and the SVM partitioning results via plots. For example:

 

6.	Compare the SVM partitioning results with the true labeling. (TP, FN, FP and TN). 
7.	Calculate the True Positive rate, False Positive Rate, Accuracy and Precision. (see the Lab_5)
8.	Open the file lab5_02.py and read the code. Simulate 2 new sets of points with two classes.  
9.	Verify the model received in 4. on the new sets. 
10.	Show the SVM partitioning results via different colors and signs like in 5. 
11.	Compare the SVM partitioning results with the true labeling. (TP, FN, FP and TN). 
12.	Calculate the True Positive rate, False Positive Rate, Accuracy and Precision.

