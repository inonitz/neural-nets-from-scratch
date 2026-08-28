import numpy as np


class Operations:
    def __init__(self, printMessage=True):
        if printMessage:
            print("Hello! Operations Lib contains: Pooling2D, Conv2D, Padding, FeedForward, ReLU ||| \n"
                  "Note: Please use dimensions (w, h, d) when inputting an image to any of the functions")

    @staticmethod
    def reLU(mat):
        return np.max(0, mat)

    @staticmethod
    # equation : (w - f + 2p) / s  +1
    def compute_spatial_size(w, f, s):
        return (w - f + 2 * (f - 1) / 2) / s + 1

    @staticmethod
    def normalize(mat, low, high):
        return

    @staticmethod
    def Padding(mat, f):
        if (f - 1) / 2 != 0 and (f - 1) % 2 == 0:
            if len(mat.shape) == 2:
                return np.pad(mat, int((f - 1) / 2), "constant")
            elif len(mat.shape) == 3:
                mat = mat.T
                return np.array([np.pad(mat[i], int((f - 1) / 2), 'constant') for i in range(mat.shape[0])]).T
        return mat

    @staticmethod
    def Stride(mat, kernel_size, stride):
        """

        :param mat: 2D Slice of the img. Helper function.
        :param kernel_size: kernel size for striding {square kernel}.
        :param stride: the amount of pixels on which the kernel will stride on.
        :return: a stridden list of matrices with kernel size kernel_size.
        """
        if len(mat.shape) > 3:
            raise Exception("Stride Helper function can only stride over 1 slice (2D) of the 3D image. Please use a for loop")

        stridden_mat = []
        for i in range(0, mat.shape[0], stride):
            if kernel_size + i <= mat.shape[0]:
                for j in range(0, mat.shape[1], stride):
                    if kernel_size + j <= mat.shape[1]:
                        stridden_mat.append(mat[i: kernel_size + i, j: kernel_size + j])

        return [stridden_mat[i].flatten() for i in range(len(stridden_mat))]

    def conv3D(self, img, filter, stride):
        convolved = []
        if type(filter) is not np.ndarray:
            raise Exception("Filters must be of type: <numpy.ndarray>!")
        elif len(filter.shape) > 4:
            raise Exception("Conv2D convolve's only with multiple filters at a time; Please use appropriate dimensions.")

        for filter in filter:
            convolved.append(self.conv2D(img, filter, stride))
        return np.array(convolved).T

    def Pooling3D(self, img, kernel_size, stride):
        if type(img) is not np.ndarray:
            raise Exception("img must be of type: <numpy.ndarray>!")
        if img.shape[0] != 4 or img.shape[0] != 3:
            raise Exception("Pooling3D Accepts image dimensions as (d, w, h)!\n If dimension depth < 3, please use pooling2D"
                            "If dimension depth > 4, please use appropriate depth (3)")
        img = img.T
        for d in range(img.shape):
            pass

    def conv2D(self, img, kernel, stride):
        """
        Dimensions of img: (w, h, d)
        :param img: Matrix Or Tensor On which the convolution will occur (Only accepts dimensions (d, w, h)
        :param kernel: The Filter used in the convolution process.
        :param stride: The amount of pixels the filter will slide in each stride
        :return: convoluted img.
        """
        w, h = int((img.shape[0] - kernel.T.shape[0] + stride) / stride), int((img.shape[1] - kernel.T.shape[1] + stride) / stride)
        if type(kernel) is not np.ndarray:
            raise Exception("Filter must be of type: <numpy.ndarray>!")
        elif len(kernel.shape) > 3:
            raise Exception("Conv2D convolve's only 1 Filter at a time! ; Please use Conv3D for multiple filters")
        # elif img.T.shape[0] != kernel.shape[0]:
        #     raise Exception("Filter and Image must have the same amount of channels (depth --> [:, :, d])\nimg.shape: {} ; filter.shape: {}".format(img.shape, kernel.shape))
        elif w % 2 != 0 or h % 2 != 0:
            raise Exception("Output dimension of Convolution(img, filter) is float!, please choose a different stride.")

        img = img.T
        convolved_mat = []
        for d in range(img.shape[0]):
            convolved_mat.append(self.Stride(img[d], kernel.T.shape[0], stride) * kernel[d].flatten())

        convolved_mat = np.array(convolved_mat)
        convolved_mat = np.sum([convolved_mat[i] for i in range(convolved_mat.shape[0])], 0)
        return np.array([np.sum(convolved_mat[i]) for i in range(convolved_mat.shape[0])]).reshape(w, h).T
        # ^
        # |
        # |
        # The Algorithm:
        # Invert Tensor from (w, h, d) to (d, w, h)
        # 1. in a for loop over all channels d:
        # 2. Multiply:
        #        Stridden Channel d Array (each stride is flattened) by the Flattened weight kernel at channel d
        #        Meaning: (Function strides over the image at channel d; we get an array of matrices (array of strides).
        #                  We flatten the matrices, THEN we Multiply Element-Wise the strides at channel d by the weight
        #                  kernel at channel d.
        #                  Note: The weight kernel and strides have the Same Dimensions!).
        #
        # 3. Sum ALL 3 convolved channels
        # 4. Sum all elements in the output filter.

    @staticmethod
    def Pooling2D(mat, kernel_size=2, stride=2):
        """
        Recommended to use kernel_size: 2, stride: 2
                           kernel_size: 3, stride: 2
        Note: Pooling2D Accepts ONLY 2D Slices of the output convolution
        """
        if (mat.shape[0] - kernel_size) % stride != 0 or (mat.shape[1] - kernel_size) % stride != 0:
            raise Exception(
                "Output dimension of Pooling(mat, kernel_size) is a float!, please choose a different stride")
        elif len(mat.shape) >= 3 or len(mat.shape) < 2:
            raise Exception("Pooling2D Accepts ONLY 2D Slices! (Matrices).")

        w, h = int((mat.shape[0] - kernel_size) / stride) + 1, int((mat.shape[1] - kernel_size) / stride) + 1
        mat = mat.T
        pooled_mat = []
        for i in range(0, mat.shape[0], stride):
            if kernel_size + i <= mat.shape[0]:
                [pooled_mat.append(np.max(mat[i: kernel_size + i, j: kernel_size + j]))
                 if kernel_size + j <= mat.shape[1]
                 else 0
                 for j in range(0, mat.shape[1], stride)
                 ]
        try:
            return np.array(pooled_mat).reshape(w, h).T
        except ValueError:
            raise Exception(
                "cannot reshape array with current stride: {}. Please try a different value for Striding".format(
                    stride))

    def feedforward(self, inputs):
        pass
