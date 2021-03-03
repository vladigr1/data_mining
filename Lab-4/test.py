from functools import *

a = [0, 1, 2, 3, 2, 1]
print( reduce(lambda i1,i2: i1 if a[i1]>a[i2] else i2,range(len(a)),0 ))