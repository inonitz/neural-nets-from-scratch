import numpy as np
from testing.Operations import Operations


def func(**kwargs):
    if 'conv2D' in kwargs and kwargs.get('conv2D'):
        print("Yes")


func(conv2D=True)


def striding(mat, kernel_size, stride):
    stridden_mat = []
    for i in range(0, mat.shape[0], stride):
        if kernel_size + i <= mat.shape[0]:
            for j in range(0, mat.shape[1], stride):
                if kernel_size + j <= mat.shape[1]:
                    stridden_mat.append(mat[i: kernel_size + i, j: kernel_size + j])

    return [stridden_mat[i].flatten() for i in range(len(stridden_mat))]


"""
Conv2D:
Convolve's Image by filter with a specific stride argument.
note: 
"""


def conv2D(img, filter, stride):
    #  Dimensions of img: (w, h, d)
    if type(filter) is not np.ndarray:
        raise Exception("Filter must be of type: <numpy.ndarray>!")
    elif img.T.shape[0] != filter.T.shape[0]:
        raise Exception("Filter and Image must have the same amount of channels (depth --> [:, :, d])")
    elif (img.shape[0] - 1) % stride != 0 or (img.shape[1] - 1) % stride != 0:
        raise Exception("Output dimension of Convolution(img, filter) is float!, please choose a different stride")

    w, h = int((img.shape[0] - filter.shape[0]) / stride + 1), int((img.shape[1] - filter.shape[1]) / stride + 1)
    img = img.T
    filter = filter.T
    convolved_mat = []
    for d in range(img.shape[0]):
        convolved_mat.append(striding(img[d], filter.shape[1], stride) * filter[d].flatten())

    convolved_mat = np.array(convolved_mat)
    convolved_mat = np.sum([convolved_mat[i] for i in range(convolved_mat.shape[0])], 0)
    return np.array([np.sum(convolved_mat[i]) for i in range(convolved_mat.shape[0])]).reshape(w, h)
    # ^
    # |
    # |
    # The Algorithm:
    # Invert Tensor from (w, h, d) to (d, w, h)
    # 1. in a for loop over all channels d:
    # 2. Multiply:
    #           Stridden Channel d Array (each stride is flattened) by the Flattened weight kernel at channel d
    #           Meaning: (Function strides over the image at channel d; we get an array of matrices (array of strides).
    #                     We flatten the matrices, THEN we Multiply Element-Wise by the weight kernel at channel d.
    #                     Note: The weight kernel and strides have the Same Dimensions!).
    #
    # 3. Sum ALL 3 convolved channels
    # 4. Sum all elements in the output filter.


op = Operations()
x = np.random.rand(5, 5, 3)
filter1 = np.random.randint(-1, 2, (5, 5, 3))
print("img", x.shape)
print("filter", filter1.shape)
x = op.Padding(x, filter1.shape[0])
print("      ", x.shape)
print(np.array(conv2D(x, filter1, 2)))
