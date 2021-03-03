# -*- coding: utf-8 -*-
"""
@author: ravros
very ugky imp more of playing with panda 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_confustion_table(data, true_tag):
    classified_matrix = np.zeros((2,2),dtype=np.double) 
    for index, row in data.iterrows():
        if row['real'] == true_tag :
            # Actual positive
            if row['pred'] == true_tag : 
                classified_matrix[0,0] = classified_matrix[0,0] + 1
            else:
                classified_matrix[0,1] = classified_matrix[0,1] + 1
        else : 
            # Actual negative
            if row['pred'] == true_tag :
                classified_matrix[1,0] = classified_matrix[1,0] + 1
            else:
                classified_matrix[1,1] = classified_matrix[1,1] + 1
    confusion_table = pd.DataFrame(classified_matrix, columns=["classified positive", " classified negative"], 
                                      index = ["actual positive", "actual negative"]) 
    return confusion_table 

def print_statistical_proababilty(confusion_table_list):
    print("True positive rate is {:f}.".format(confusion_table_list[0,0] / (confusion_table_list[0,0] + confusion_table_list[0,1])))
    print("False positive rate is {:f}.".format(confusion_table_list[1,0] / (confusion_table_list[1,1] + confusion_table_list[1,0])))
    print("Accuracy is {:f}.".format(( confusion_table_list[0,0] + confusion_table_list[1,1]) / (confusion_table_list[0,0] + confusion_table_list[0,1] + confusion_table_list[1,0] + confusion_table_list[1,1])))
    print("Precision is {:f}.".format(confusion_table_list[0,0] / (confusion_table_list[0,0] + confusion_table_list[1,0])))

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)
irisdata = irisdata[50:150]

data = irisdata.drop('Class', axis=1)
real_label = irisdata['Class']
from sklearn.model_selection import train_test_split
data_train, data_test, real_label_train, real_label_test = train_test_split(data, real_label, test_size = 0.20)

from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(data_train, real_label_train)
pred_label_test = svclassifier.predict(data_test) # partitioning result label

true_tag = "Iris-versicolor"
colors = ['#1f77b4', '#ff7f0e']
data_train = data_train.assign(pred=[ element for element in real_label_train ])
data_train = data_train.assign(real=[ real_label_train[i] for i in data_train.index ])
data_train = data_train.assign(color=[(colors[1] if real_label[i] == true_tag else colors[0]) for i in data_train.index])
plt_data_pred_is_true_tag = data_train.where(data_train["pred"] == true_tag).dropna(subset=["pred"])
plt.scatter(plt_data_pred_is_true_tag["sepal-length"],plt_data_pred_is_true_tag["sepal-width"],c=plt_data_pred_is_true_tag["color"],marker="x",label = "train set real_label=true")
plt_data_pred_not_true_tag = data_train.where(data_train["pred"] != true_tag).dropna(subset=["pred"])
plt.scatter(plt_data_pred_not_true_tag["sepal-length"],plt_data_pred_not_true_tag["sepal-width"],c=plt_data_pred_not_true_tag["color"],marker="+",label = "train set real_label=false")
# data test
data_test = data_test.assign(pred=[ element for element in pred_label_test ])
data_test = data_test.assign(real=[ real_label_test[i] for i in data_test.index ])
data_test = data_test.assign(color=[(colors[1] if real_label[i] == true_tag else colors[0]) for i in data_test.index])
plt_data_pred_is_true_tag = data_test.where(data_test["pred"] == true_tag).dropna(subset=["pred"])
plt.scatter(plt_data_pred_is_true_tag["sepal-length"],plt_data_pred_is_true_tag["sepal-width"],c=plt_data_pred_is_true_tag["color"],marker="1",label = "test set pred=true")
plt_data_pred_not_true_tag = data_test.where(data_test["pred"] != true_tag).dropna(subset=["pred"])
plt.scatter(plt_data_pred_not_true_tag["sepal-length"],plt_data_pred_not_true_tag["sepal-width"],c=plt_data_pred_not_true_tag["color"],marker="^",label = "test set pred=false")
plt.legend()
plt.show()
confusion_table = generate_confustion_table(data_test, true_tag)
print(confusion_table)
print_statistical_proababilty(confusion_table.values)


# blob dup Iris but ugly beacuse wanted to feel the panda lib by hand
n_samples = 100
true_tag = 0
from sklearn.datasets.samples_generator import make_blobs
X, X_real_label = make_blobs(n_samples=n_samples, centers=2,cluster_std = 0.30, random_state = 5)
data_train, data_test, real_label_train, real_label_test = train_test_split(X, X_real_label, test_size = 0.20)

svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(data_train, real_label_train)
pred_label_test = svclassifier.predict(data_test) # partitioning result label

data_train = pd.DataFrame(data_train)
real_label_train = pd.DataFrame(real_label_train)

data_test= pd.DataFrame(data_test)
real_label_test= pd.DataFrame(real_label_test)

data_train = data_train.assign(pred=[ element for element in real_label_train[0] ])
data_train = data_train.assign(real=[ real_label_train[0][i] for i in data_train.index ])
data_train = data_train.assign(color=[(colors[1] if real_label_train[0][i] == true_tag else colors[0]) for i in data_train.index])
plt_data_pred_is_true_tag = data_train.where(data_train["pred"] == true_tag).dropna(subset=["pred"])
plt.scatter(plt_data_pred_is_true_tag[0],plt_data_pred_is_true_tag[1],c=plt_data_pred_is_true_tag["color"],marker="x",label = "train set real_label=true")
plt_data_pred_not_true_tag = data_train.where(data_train["pred"] != true_tag).dropna(subset=["pred"])
plt.scatter(plt_data_pred_not_true_tag[0],plt_data_pred_not_true_tag[1],c=plt_data_pred_not_true_tag["color"],marker="+",label = "train set real_label=false")
# data test
data_test = data_test.assign(pred=[ element for element in pred_label_test ])
data_test = data_test.assign(real=[ real_label_test[0][i] for i in data_test.index ])
data_test = data_test.assign(color=[(colors[1] if real_label_test[0][i] == true_tag else colors[0]) for i in data_test.index])
plt_data_pred_is_true_tag = data_test.where(data_test["pred"] == true_tag).dropna(subset=["pred"])
plt.scatter(plt_data_pred_is_true_tag[0],plt_data_pred_is_true_tag[1],c=plt_data_pred_is_true_tag["color"],marker="1",label = "test set pred=true")
plt_data_pred_not_true_tag = data_test.where(data_test["pred"] != true_tag).dropna(subset=["pred"])
plt.scatter(plt_data_pred_not_true_tag[0],plt_data_pred_not_true_tag[1],c=plt_data_pred_not_true_tag["color"],marker="^",label = "test set pred=false")
plt.legend()
plt.show()
confusion_table = generate_confustion_table(data_test, true_tag)
print(confusion_table)
print_statistical_proababilty(confusion_table.values)