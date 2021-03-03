# Lab 8: Image processing
we will have 

## Python - new things
### Lab 8_0
* `zip(digits.images, digits.target)` 
  zip(*iterables) --> A zip object yielding tuples until an input is exhausted.
  ```python
    list(zip('abcdefg', range(3), range(4)))
    [('a', 0, 0), ('b', 1, 1), ('c', 2, 2)]
  ```
* `ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')`
  image, gray , pixel will describe to nearest values.
* `data = digits.images.reshape((n_samples, -1))`
  reshape the 2D image to 64 index vector.
* `n_samples // 2` == float division.

* `classifier = svm.SVC(gamma='scale')`
    gamma : {'scale', 'auto'} or float, default='scale'\
    Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.\

    - if ``gamma='scale'`` (default) is passed then it uses
        1 / (n_features * X.var()) as value of gamma,
    - if 'auto', uses 1 / n_features.

## Tasks to do
1.	Open the files Lab_8_0_SVM.py.  
2.	Load the Digits dataset digits = datasets.load_digits()  The dataset consists of 1797 gray-level (0-16) images of size 8X8 pixels. 
3.	Choose a half of the images as a Training Set and rest of the images as a Testing Set.
4.	Construct a support vector classifier on the Training Set.
5.	Predict the value of the digit on the Testing Set.
=============================================================
6.	Open the files Lab_8_1.py.  
7.	Take the image sad_cat.jpg 
8.	Take the file titles.jpg, convert it to grayscale and make a negative of the picture (instead of dark text on a white background, we obtain light text on a black background).
9.	Crop automatically one of the titles (in English or in Hebrew), and replace the title in Russian in the sad_cat.jpg image with the title in other language as shown on the image bellow. 
