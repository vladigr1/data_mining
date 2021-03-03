#from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt 
# graph theme
import seaborn as sns; sns.set()
#import pandas as pd 
from pandas import DataFrame
from sklearn.metrics import silhouette_samples, silhouette_score
#import datasets 
from sklearn import datasets
# Generate some data
from sklearn.datasets.samples_generator import make_blobs

def clust(df,n_cl):
    kmeans = KMeans(n_clusters = n_cl).fit(df)
    res = kmeans.labels_+1
    centroids = kmeans.cluster_centers_

    plt.scatter(df['x1'], df['x2'], c = res, s=50, alpha = 1)

    # add center (center is 4D vector)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.title("scatter x2 vs x1")
    plt.show()

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(range(len(res)),res)
    plt.title("bar graph for labels")
    plt.show()
#show the silhouette values for k=3 
    silhouette_avg_ = silhouette_score(df, res)
    sample_silhouette_values_ = silhouette_samples(df, res)  
    plt.plot(sample_silhouette_values_) 
    plt.plot(silhouette_avg_, 'r--')
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
    y=silhouette_avg_
    xmin=0
    xmax=len(res)
# The vertical line for average silhouette score of all the values
    plt.hlines(y, xmin, xmax, colors='red', linestyles="--") 
    plt.show()

    print("For n_clusters =", n_cl,
      "The average silhouette_score is:", silhouette_avg_)
    return res

def generate_confustion_table(real_label, kmeans_labels,n_samples, positive_flower_label):
    classified_matrix = np.zeros((2,2),dtype=np.double) 
    for i in range(0,n_samples):
        if real_label[i] == positive_flower_label :
            # Actual positive
            if kmeans_labels[i] == positive_flower_label : 
                classified_matrix[0,0] = classified_matrix[0,0] + 1
            else:
                classified_matrix[0,1] = classified_matrix[0,1] + 1
        else : 
            # Actual negative
            if kmeans_labels[i] == positive_flower_label :
                classified_matrix[1,0] = classified_matrix[1,0] + 1
            else:
                classified_matrix[1,1] = classified_matrix[1,1] + 1
    confusion_table = DataFrame(classified_matrix, columns=["classified positive", " classified negative"], 
                                      index = ["actual positive", "actual negative"]) 
    return confusion_table 

def print_statistical_proababilty(confusion_table_list):
    print("True positive rate is {:f}.".format(confusion_table_list[0,0] / (confusion_table_list[0,0] + confusion_table_list[0,1])))
    print("False positive rate is {:f}.".format(confusion_table_list[1,0] / (confusion_table_list[1,1] + confusion_table_list[1,0])))
    print("Accuracy is {:f}.".format(( confusion_table_list[0,0] + confusion_table_list[1,1]) / (confusion_table_list[0,0] + confusion_table_list[0,1] + confusion_table_list[1,0] + confusion_table_list[1,1])))
    print("Precision is {:f}.".format(confusion_table_list[0,0] / (confusion_table_list[0,0] + confusion_table_list[1,0])))

if __name__ == "__main__":
    # Iris data set
    start_data_index, end_data_index = 50,150
    Iris = datasets.load_iris()
    iris_data = Iris.data 
    real_label = Iris.target[50:150] # true labeling

    y0 = np.array(iris_data[:,0:4])
    y2 = np.array(iris_data[start_data_index:end_data_index,0:4])
    df = DataFrame(y2, columns=['x1','x2','x3','x4']) 
    kmeans_labels = clust(df,2)

    positive_flower_label = 1
    confusion_table  = generate_confustion_table(real_label, kmeans_labels, end_data_index-start_data_index, positive_flower_label)
    print("positive flowe label is : {:d}".format(positive_flower_label))
    print(confusion_table)
    print_statistical_proababilty(confusion_table.values)

    # Blob data
    n_samples = 200
    X, real_label = make_blobs(n_samples=n_samples, centers=2,
                       cluster_std = 0.90, random_state = 5)
    real_label = real_label + 1
    kmeans = KMeans(2, random_state = 5)
    kmeans_labels = kmeans.fit(X).predict(X) + 1
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, s=40, cmap='viridis');
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.title("blob graph")
    plt.show
    positive_blob_label = 1
    confusion_table  = generate_confustion_table(real_label, kmeans_labels, n_samples, positive_blob_label)
    print("positive blob label is : {:d}".format(positive_blob_label))
    print(confusion_table)
    print_statistical_proababilty(confusion_table.values)