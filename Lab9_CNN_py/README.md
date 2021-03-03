# CNN Lab 9: Convolutional Neural Networks
## Python - new things
### Lab 8_0
* `zip(digits.images, digits.target)` 
  zip(*iterables) --> A zip object yielding tuples until an input is exhausted.
  * `
  ```python
    list(zip('abcdefg', range(3), range(4)))
    [('a', 0, 0), ('b', 1, 1), ('c', 2, 2)]
  ```
* `X_test.shape` --> get dimensions of numpy array.
* ```python
  y_test = np_utils.to_categorical(y_test)
  lambda_return_to_value = lambda y_i: argmax(y_i, axis=None, out=None)` g
  ```
  you must change tag to vector to return in use this lambda expression
## Tasks to do
1.	Read the file CNN_Lab_9. Open the files lab_9_0.py and lab_9_1.py .  
2.	Load the MNIST Database of handwritten digits from Keras:  keras.datasets import mnist
3.	Choose a part of the images as a Training Set and rest of the images as a Testing Set.
4.	Define a simple CNN model def baseline_model()
5.	Construct the model on the Training Set.
6.	Predict the value of the digit on the Testing Set.
Independent work
7.	Draw on a separate piece of paper (or in Paint) digit '8'.  Use smartphone for scanning this picture and create the image file. You can use Paint to draw and create the image file. 
 
8.	Crop from the considered image up to the format of MNIST Database. Resize the image to size 28x28. 
9.	Use CNN algorithm to build a model – the trained network for the MNIST Database.
10.	Predict the labels of your image using the trained network.
11.	Show your image with it's prediction result.
12.	Repeat the steps 7-8 for digit '7'.
13.	Use CNN algorithm to build the trained network for a subset from MNIST Database (classes of digits '1' and '7').
14.	Predict the labels of your '7' image using the trained network from step 13.
15.	Show your image with it's prediction result. 

