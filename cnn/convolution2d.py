import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# from skimage import color


def plot(mat, gray=False):
    if gray:
        plt.imshow(mat, cmap=plt.gray())
    else:
        plt.imshow(mat)
    plt.show()


# Helper function
def rot180(mat):
    return np.flipud(np.fliplr(mat))


def pad3D(mat, kernel_size, stride, di):
    if (kernel_size - 1) % stride == 0:
        return np.array([pad2D(mat[:, :, i], kernel_size, stride) for i in range(mat.shape[di])]).T
    raise Exception("cant pad; padding constant is a float value!")


def pad2D(mat, kernel_size, stride):
    if (kernel_size - 1) % stride == 0:
        return np.pad(mat, int((kernel_size - 1) / stride), 'constant').T
    raise Exception("cant pad; padding constant is a float value!")


def convolve3D(mat, kfilter, stride):
    kfilter = rot180(kfilter)
    return [
        [np.sum(mat[i: i + kfilter.shape[0], j: j + kfilter.shape[1], :] * kfilter) for j in range(0, mat.shape[1], stride)
         if j + kfilter.shape[1] <= mat.shape[1]] for i in range(0, mat.shape[0], stride)
        if kfilter.shape[0] + i <= mat.shape[0]]


def pooling2D(mat, kernel_size, stride):
    return np.array([
        [np.max(mat[i: i + kernel_size, j: j + kernel_size]) for j in range(0, mat.shape[1], stride)
         if j + kernel_size <= mat.shape[1]] for i in range(0, mat.shape[0], stride)
        if kernel_size + i <= mat.shape[0]])


img2 = np.random.randint(-1, 2, (5, 5, 3))
itg = np.array([[3, 9, 5, 9],
                [1, 7, 4, 3],
                [2, 1, 6, 5]])

kernel_mat = np.array([[1, 0, -1],
                       [0, 0, 0],
                       [-1, 0, 1]])

kernelmat = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]])

sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

img = Image.open('Vd-Orig.png')
img.load()
img = np.array(img)


filter = np.array([sharpen for i in range(img.shape[-1])]).T
vstride = 1
pstride = 2
kernel_osize = filter.shape[0]
# filter2 = np.random.rand(3, 3, 4)

padded = pad3D(img, kernel_osize, vstride, -1)
y = np.array(convolve3D(padded, filter, vstride))
pooled = pooling2D(y, kernel_osize, pstride)

plot(img)
plot(padded)

print("For padding, convolution stride {} | pooling stride {} | && kernel_size {}".format(vstride, pstride, kernel_osize))
print("img shape [padded | not] {} {} | convolution shape {} | filter shape {} | pooled shape {}"
      .format(padded.shape, img.shape, y.shape, filter.shape, pooled.shape))

plot(y, gray=True)
plot(pooled, gray=True)
