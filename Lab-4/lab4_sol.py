import re
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_samples, silhouette_score


def readFile(fileName):
    file = open(fileName,'r')
    fileStr = ""
    for line in file:
        fileStr += line
    return fileStr
    
def preProcess(fileStr):
    fileStr = re.sub(" +"," ", fileStr)
    fileStr = re.sub("[^a-zA-Z ]","", fileStr)
    fileStr = fileStr.lower()
    return fileStr

#finction clust
def clust(dist,n_cl):
#cluster the data into k clusters, specify the k  
    kmeans = KMeans(n_clusters = n_cl)
    kmeans.fit(dist)
    labels = kmeans.labels_ + 1
#show the clustering results  
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(range(len(labels)),labels)
    plt.show()

# calculate the silhouette values  
    silhouette_avg_ = silhouette_score(dist, labels)
    sample_silhouette_values_ = silhouette_samples(dist, labels)
# show the silhouette values 
    # plt.plot(sample_silhouette_values_) 
    # plt.plot(silhouette_avg_, 'r--')
    # plt.title("The silhouette plot for the various clusters.")
    # plt.xlabel("The silhouette coefficient values")
    # plt.ylabel("Cluster label")
    # y=silhouette_avg_
    # xmin=0
    # xmax=len(labels)
# The vertical line for average silhouette score of all the values
    # plt.hlines(y, xmin, xmax, colors='red', linestyles="--") 
    # plt.show()
    print("For n_clusters =", n_cl,
      "The average silhouette_score is:", silhouette_avg_)
    return sample_silhouette_values_, silhouette_avg_, labels


#Divide the file in chuncks of the same size wind
def partition_str(fileStr, wind):
    n = wind
    chunks = [fileStr[i:i+n] for i in range(0, (len(fileStr)//n)*n, n)]
    #print(chunks)
    count = len(chunks)
    return chunks, count;

def createWordDictionary(allFileStr):
    wordSet = set(allFileStr.split())
    # Read stop words file - words that can be removed
    stopWordsSet = set(readFile('stopwords_en.txt').split())
    # Remove the stop words from the word list
    return wordSet

def findBestText(silhouette_avg_list, label_list):
    index = silhouette_avg_list.index(max(silhouette_avg_list))
    curLabel = label_list[index][0]
    curchar = 0
    for i,label in enumerate( label_list[index] ):
        if label != curLabel:
            print("from char {:d} to {:d}".format(curchar*5000, i*5000))
            curchar = i
            curLabel = label
    print("from char {:d} to {:d}".format(curchar*5000, i*5000))

if __name__ == "__main__":
   wind = 5000
   fileStr = preProcess(readFile("text1.txt")) 
   chunks, count  = partition_str(fileStr,wind)
   dictionary= createWordDictionary(fileStr)
   # Count the number of dictionary words in files - Frequency Matrix
   wordFrequency = np.empty((count,len(dictionary)),dtype=np.int64)
   for i in range(count):
       print(i)
       for j,word in enumerate(dictionary):
           wordFrequency[i,j] = len(re.findall(word,chunks[i]))
           
   # find the distance matrix between the text files - Distance Matrix
   dist = np.empty((count,count))
   for i in range(count): 
       for j in range(count):
           # calculate the distance between the frequency vectors
           dist[i,j] = np.linalg.norm(wordFrequency[i,:]-wordFrequency[j,:])
   y2, silhouette_avg2, label2 = clust(dist,2)
   y3, silhouette_avg3, label3= clust(dist,3)
   y4, silhouette_avg4, label4= clust(dist,4)
   fig, axs = plt.subplots(3)
   axs[0].plot(range(count),y2)
   axs[0].set_title('k = 2')
   axs[1].plot(range(count),y3)
   axs[1].set_title('k = 3')
   axs[2].plot(range(count),y4)
   axs[2].set_title('k = 4')
   plt.show()
   findBestText([silhouette_avg2, silhouette_avg3, silhouette_avg4], [label2, label3, label4]) 