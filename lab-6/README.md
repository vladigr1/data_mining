# Lab 6: K-Nearest Neighbor KNN algorithm
Supervised Learning- you know the tag but with the vector elements of the object find common feature for same tags.\
With this communality of traits we classified other data.

Nearest neighbor method - the common feature will be vector distance for the tag values.

Distance is abrutarly you can choose the best norm.\
distance can be weighted\
Commonly the weight is 1/d^2

Conflicts , k is number points which are nearest to current point:\
if k is too small it will be sensitive to noise point.\
if k is too big, it will average the result, which means you lose the point using this method.

## Python - new things
* `np.shape(data)` == MATALB [m,n] = size(data);
* `round(num)` = round number
* `TrainingSet = np.concatenate((x1,x2),axis=0)` == like append but for np.array
  axis =0 , add as row
  axis =1 , add as columns
  ```python
  a = np.array([[1, 2], [3, 4]])
  b = np.array([[5, 6]])
  np.concatenate((a, b), axis=0)
  array([[1, 2],
         [3, 4],
         [5, 6]])
  
  np.concatenate((a, b.T), axis=1)
  array([[1, 2, 5],
         [3, 4, 6]])
  ```

* `np.array([[5, 6]])np.array([[5, 6]]).T` == transpose()

* `heapq.heapify(x)` == Transform list x into a heap, in-place, in linear time.\
  `heapq.nlargest(n, iterable, key=None)`  == n largest elements\
  `heapq.nsmallest(n, iterable, key=None)` == n largest elements

## Tasks to do:
1.	Open the file Lec_KNN.ppt and read description of KNN algorithm.
2.	Open the file lab6_ex1.py. 
3.	Load the data from Iris Database.
4.	Choose 20 points from each group (see species array) as a Training Dataset. 
5.	For a new point (test_row) find the nearest point from the Training Dataset.

Independent work
1.	Compose the Testing Dataset from the points not chosen for the Training Dataset.
2.	Apply KNN algorithm for K=1 to the Testing Dataset.  Show results in a visual way.
3.	Implement KNN algorithm to divide Testing Dataset into two groups for arbitrary K.


