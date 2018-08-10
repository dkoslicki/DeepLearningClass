# jupyter.org/try

# Workshop examples
import numpy as np
import pandas as pd
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

# Pandas!

data = np.array([['', 'A', 'B', 'C'],
                 ['Row0', 1, 2, 3],
                 ['Row1', 3, 4, 5],
                 ['Row2', 6, 7, 8]])


df = pd.DataFrame(data=data[1:, 1:],
                  index=data[1:, 0],
                  columns=data[0, 1:])

# alternatively, to create row-wise
cols = {"A": [1, 3, 6],
		"B": [2, 4, 7],
		"C": [3, 5, 8]}

df = pd.DataFrame(data=cols)

# grabbing values
print(df.iloc[0][0])

# another way to access stuff
df.loc[:, 'B']
df.loc[:, 'C']

df['A']
df.to_csv('test.csv')

df2 = pd.read_csv('test.csv')


