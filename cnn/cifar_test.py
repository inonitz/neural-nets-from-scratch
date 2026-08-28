import numpy as np
from operate import operations
import pickle
from PIL import Image


def unpickle(pf):
    with open(pf, 'rb') as upf:
        dicti = pickle.load(upf, encoding='bytes')
    return dicti


def get_inputs(data):
    return np.array([np.rot90(np.fliplr(data[i].reshape(3, 32, 32).T)) for i in range(data.shape[0])])


x = operations(msg=False)
convnet_arch = {'[ [ Conv -> ReLU ] * 2 --> Pool ]*2 --> FC --> ReLU --> FC'}
img = unpickle('data_batch_1')
img2 = Image.open("wha.jpg")
img2.load()

print(type(img))
[print(item, type(value)) for item, value in img.items()]

# arr = get_inputs(img[b'data'])
arr = np.array(img2)
labels = img[b'labels']
filenames = img[b'filenames']
print(len(filenames), '\n\n')



kernelmat = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]])


sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])


kernel_mat = np.array([[1, 0, -1],
                       [0, 0, 0],
                       [-1, 0, 1]])


stride = 1
pool_fs = (2, 2)
pool_s = 2
dim = 3

filter = np.array([kernelmat for i in range(dim)])
filter2 = np.array([sharpen for it in range(dim)])
filter3 = np.array([kernel_mat for ite in range(dim)])

x.plot(arr)
# [x.plot(arr[i]) for i in range(5)]


conv1 = x.convolve3D(arr, filter, stride)
conv2 = x.convolve3D(arr, filter2, stride)
conv3 = x.convolve3D(arr, filter3, stride)


r1 = x.reLU(conv1)
r2 = x.reLU(conv2)
r3 = x.reLU(conv3)

pool1 = x.pooling2D(r1, pool_fs[0], pool_s)
pool2 = x.pooling2D(r2, pool_fs[0], pool_s)
pool3 = x.pooling2D(r3, pool_fs[0], pool_s)


#  Keep going with sharpen filter until all features extracted
#  3 Weight filters in next layer applied to output of  pool2 /w filter123
conv4 = x.convolve2D(pool2, filter[:, :, 0], stride)
conv5 = x.convolve2D(pool2, filter2[:, :, 0], stride)
conv6 = x.convolve2D(pool2, filter3[:, :, 0], stride)


r4 = x.reLU(conv4)
r5 = x.reLU(conv5)
r6 = x.reLU(conv6)


pool4 = x.pooling2D(r4, pool_fs[0], pool_s)
pool5 = x.pooling2D(r5, pool_fs[0], pool_s)
pool6 = x.pooling2D(r6, pool_fs[0], pool_s)

print(conv2.shape, '\n', r2.shape, '\n', pool2.shape, '\n', conv5.shape,
      '\n', r5.shape, '\n', pool5.shape)
x.plot(conv2)
x.plot(r2)
x.plot(pool2)
x.plot(conv6)
# [[print(conv5[i][j]) for j in range(conv5.shape[1]) if conv5[i][j] > 0] for i in range(conv5.shape[0])]
x.plot(r6)
x.plot(pool6)
pool7 = x.pooling2D(pool6, pool_fs[0], pool_s)
pool8 = x.pooling2D(pool7, pool_fs[0], pool_s)
pool9 = x.pooling2D(pool8, pool_fs[0], pool_s)
pool10 = x.pooling2D(pool9, pool_fs[0], pool_s)

print(pool7.shape, pool8.shape, pool9.shape, pool10.shape)
x.plot(pool7)
x.plot(pool8)
x.plot(pool9)
x.plot(pool10)

