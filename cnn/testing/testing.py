import numpy as np
from testing.Operations import Operations
import matplotlib.pyplot as plt


def createWeight_Filter(kernel_amount, kernel_size, channels):
    return np.random.rand(kernel_amount, kernel_size, kernel_size, channels)


def plot(mat):
    plt.imshow(mat, cmap='gray')
    plt.show()


Op = Operations()
w = 25
h = 25
d = 3
f = 3
s = 2
x = np.random.randint(0, 256, (w, h))

print("Output Size should be {}, {}".format((w - f) / s + 1, (h - f) / s + 1))
# print("Time Taken for Pooling2D: {}".format(timeit.timeit("from Operations import Operations; import numpy as np; Op=Operations(); f = 2; s = 2; x = np.random.randint(0, 256, (100, 100)); Op.Pooling2D(x, f, s)", number=100)))
res = Op.Pooling2D(x, f, s)
print(res.shape, x.shape)
plot(x)
plot(res)
"""
 To generate Filter Weights: np.random.randint(-1, 2, (k, d, f, f))
 1. k --> number of filters. [output volume]
 2. d --> depth // dimensions of input [:, :, d]
 3. f --> filter size [extent] (f, f)
 4.
 ex: for filters k[i=2] , each with d[j=3] elements, each element[j] is a (3,3) matrix with size (f, f)[k]
 tensor shape: [i=2, j=3, f=3, f=3]
        [[[[ 0  0  1]
           [-1  1  0]
           [ 1 -1 -1]]

          [[ 1 -1 -1]
           [ 0 -1 -1]
           [-1  0  1]]

          [[ 1  0  1]
           [-1  1  0]
           [ 0  1  0]]],


         [[[-1  0  1]
           [-1  0 -1]
           [-1  1  0]]

          [[ 0  0  0]
           [ 1 -1  1]
           [ 1 -1  0]]

          [[-1  1 -1]
           [ 0  1 -1]
           [ 0 -1 -1]]]
"""

"""
Pooling --> Works
Convolve --> Works
feed-forward --> 
back-propagation -->

"""
