from math import sqrt
import numpy as np
#import datasets 
from sklearn import datasets
import math
import heapq
import matplotlib.pyplot as plt
from functools import reduce

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

def generate_training_set(data, true_label, index_dict ):
    # num of index per cluster must be symatrical
    num_features = np.shape(data)[1]
    label_index = num_features
    num_features_plus_true_label = num_features + 1
    size_of_single_train_cluster = index_dict['end_index_cluster1'] - index_dict['start_index_cluster1']
    x1 = np.zeros((size_of_single_train_cluster, num_features_plus_true_label ))
    x2 = np.zeros((size_of_single_train_cluster, num_features_plus_true_label ))
    x1[0:size_of_single_train_cluster, 0:num_features]= data[index_dict['start_index_cluster1']:index_dict['end_index_cluster1'], :]
    x2[0:size_of_single_train_cluster,0:num_features]=data[index_dict['start_index_cluster2']:index_dict['end_index_cluster2'], :]
    # put labels - supervised learning
    x1[0:size_of_single_train_cluster, label_index]=true_label[index_dict['start_index_cluster1']:index_dict['end_index_cluster1']]
    x2[0:size_of_single_train_cluster, label_index]=true_label[index_dict['start_index_cluster2']:index_dict['end_index_cluster2']]
    return np.concatenate((x1,x2),axis=0)

def K_nearest_classified(k, test_row, training_set):
    # expect tags [a:b] possitive numbers : beacause of counter_table
    get_label = lambda item:item[4]
    my_dist = [(euclidean_distance(test_row,item),i,get_label(item)) for i,item in enumerate(training_set)]
    nearest_k_points = heapq.nsmallest(k,my_dist)
    get_label = lambda item:item[2]
    nearest_k_points_tags = [int(get_label(point)) for point in nearest_k_points]
    counter_table = [0] * int( max(nearest_k_points_tags) + 1 )
    for i in nearest_k_points_tags:
        counter_table[i] = counter_table[i] + 1
    find_max_index_in_counter_table_func = lambda index1 , index2: index1 if counter_table[ index1 ] > counter_table[ index2 ] else index2
    max_index_in_counter_table = reduce( find_max_index_in_counter_table_func, range(0,len(counter_table)),0)
    return max_index_in_counter_table
    # bad solution too sepcific for this case
    # return round(sum(nearest_k_points_tags)/k)

def display_result(training_set, testing_set):
    plt.scatter(training_set[:, 0],training_set[:, 1],c=training_set[:, 4],label = "training set")
    plt.scatter(testing_set[:, 0],testing_set[:, 1],c=testing_set[:, 4],marker="*",label = "testing set")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #import datasets 
    index_training_dict = {}
    index_training_dict[ "start_index_cluster1" ] = 0
    index_training_dict[ "start_index_cluster2" ] =  50
    index_training_dict[ "end_index_cluster1" ] = 20
    index_training_dict[ "end_index_cluster2" ] =  70
    Iris = datasets.load_iris()
    start_dataset_index = 0
    end_dataset_index = 100
    data = Iris.data[start_dataset_index:end_dataset_index ,:]
    true_label = Iris.target[start_dataset_index:end_dataset_index ] 
    training_set = generate_training_set(data, true_label, index_training_dict)

    index_testing_dict = {}
    index_testing_dict[ "start_index_cluster1" ] = 20
    index_testing_dict[ "start_index_cluster2" ] =  70
    index_testing_dict[ "end_index_cluster1" ] = 50
    index_testing_dict[ "end_index_cluster2" ] =  100
    testing_set = generate_training_set(data, true_label, index_testing_dict)

    # classify testing set
    for row in testing_set:
        classfied_label  = K_nearest_classified(3,row[0:5], training_set)
        row[4] = classfied_label

    display_result(training_set,testing_set)

