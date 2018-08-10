# jupyter.org/try

# Workshop examples
import numpy as np

# vectors
a = np.array([1, 2, 3, 4], float)
print(a)

# matrices
a = np.array([[1, 2], [3, 4]])

2 in a  # I didn't know that! Probably O(#(a)) though
a.shape
a.dtype
a = a.reshape(1, 4)

# get values
a[0, 1]

# slicing
a[:, 1]

# np.zeros, np.ones, etc.