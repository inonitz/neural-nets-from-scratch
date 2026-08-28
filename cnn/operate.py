import numpy as np
import matplotlib.pyplot as plt


class operations:
    def __init__(self, msg=True):
        if msg:
            print("Using Operate class!")

    @staticmethod
    def comp_shape(mat_shape, filter_size, stride):
        w, h = (mat_shape[0] - filter_size[0] + stride) / stride, (mat_shape[1] - filter_size[1] + stride) / stride
        if w * stride % stride == 0 and w * stride % stride == 0:
            print(w, h)
            return int(w), int(h)
        raise Exception("result shape has floating point values. please use different parameters! [{}, {}]".format(w, h))


    @staticmethod
    def dstack(mat, amount):
        if len(mat.shape) > 2:
            raise Exception("Function only works with matrices")

        zeros = np.zeros(mat.shape)
        for i in range(amount):
            zeros = np.dstack((zeros, mat))
        return zeros[:, :, 1:]

    @staticmethod
    def stack(mat, matrices):
        for m in matrices:
            mat = np.dstack((mat, m))
        return mat

    @staticmethod
    def plot(mat, gray=False):
        if gray:
            plt.imshow(mat, cmap='gray')
        else:
            plt.imshow(mat)
        plt.show()

    @staticmethod
    def reLU(mat):
        return np.maximum(0, mat)

    @staticmethod
    def relu_prime(mat):
        mat[mat <= 0] = 0
        mat[mat > 0] = 1
        return mat

    @staticmethod
    def softmax(mat):
        return np.exp(mat) / np.sum(np.exp(mat))

    @staticmethod
    def hot_vector(label, ln):
        return np.array([0 if i != label else 1 for i in range(ln)])

    @staticmethod
    # Helper function
    def rot180(mat):
        return np.flipud(np.fliplr(mat))

    @staticmethod
    def pad2D(mat, kernel_size, stride):
        if (kernel_size - 1) % stride == 0:
            return np.pad(mat, int((kernel_size - 1) / stride), 'constant')
        raise Exception("cant pad; padding constant is a float value!")

    @staticmethod
    def pooling2D(mat, kernel_size, lstride):
        return np.array([
            [np.max(mat[i: i + kernel_size, j: j + kernel_size]) for j in range(0, mat.shape[1], lstride)
             if j + kernel_size <= mat.shape[1]] for i in range(0, mat.shape[0], lstride)
            if kernel_size + i <= mat.shape[0]])



    def convolve3D(self, mat, kfilter, stride):
        if len(kfilter.shape) == 2:
            kfilter = self.dstack(kfilter, mat.shape[-1])
        kfilter = self.rot180(kfilter)
        return np.array([
            [np.sum(mat[i: i + kfilter.shape[0], j: j + kfilter.shape[1], :] * kfilter) for j in
             range(0, mat.shape[1], stride)
             if j + kfilter.shape[1] <= mat.shape[1]] for i in range(0, mat.shape[0], stride)
            if kfilter.shape[0] + i <= mat.shape[0]])



    def pooling3D(self, tens, kernel_size, stride):
        return self.stack(self.pooling2D(tens[0], kernel_size, stride), [self.pooling2D(tens[i], kernel_size, stride) for i in range(1, tens.shape[-1])])



    def pad3D(self, tens, ks, stride):
        if (ks - 1) % stride != 0:
            raise Exception("cant pad; padding constant is a float value!")
        elif (ks - 1) % stride == 0:
            return self.stack(self.pad2D(tens[:, :, 0], ks, stride), [self.pad2D(tens[:, :, i], ks, stride) for i in range(1, tens.shape[-1])])



    # Convolve & pool & relu all together.
    def propagate_convolution(self, image, filter, stride, filter_amount=16, pool=False, poolKS=(3, 3), pStride=2, pad=False):
        """
                Disclaimer! : wfilters is the amount of feature maps wanted,
                              you may need to add param which specifies how many
                              feature maps the user wants, create the wfilters[amount=feature_amount],
                              get convolved result, reLU, pool(maybe) && return WFILTERS && RESULT!!!

                Convolve feature maps // image using stride && wfilters.
                ReLU activation.

                pooling layer (optional, add option to add layer of pooling)
                if pooling, enter kernel_size (poolKS) && stride (pool_stride)
                pool the convolved feature maps

                return reLU'd, convolved, filter[optional].

        """
        convolved = []
        if type(filter) is tuple or type(filter) is list:
            filter = np.random.normal(0, np.sqrt(2. / image.shape[-1] * filter[0] * filter[1]),
                                      (filter[0], filter[1], filter_amount))
        if pad:
            image = self.pad3D(image, filter.shape[0], stride)

        for i in range(filter_amount):
            convolved = self.stack(self.convolve3D(image, filter[:, :, 0], stride), [self.convolve3D(image, filter[:, :, i], stride) for i in range(1, filter_amount)])


        if not pool:
            return self.reLU(convolved), filter

        return self.reLU(self.pooling3D(tens=convolved, kernel_size=poolKS, stride=pStride)), convolved, filter



    def convolve(self, image, filter, stride, filter_amount=16, pad=False):
        convolved = []
        if type(filter) is tuple or type(filter) is list:
            filter = np.random.normal(0, np.sqrt(2. / image.shape[-1] * filter[0] * filter[1]),
                                      (filter[0], filter[1], filter_amount))
        if pad:
            image = self.pad3D(image, filter.shape[0], stride)

        for i in range(filter_amount):
            convolved = self.stack(self.convolve3D(image, filter[:, :, 0], stride),
                                   [self.convolve3D(image, filter[:, :, i], stride) for i in range(1, filter_amount)])
        return convolved, filter


    def pool(self, conv, poolKS=(3, 3), pStride=2):
        return self.pooling3D(tens=conv, kernel_size=poolKS, stride=pStride)


    def feedforward(self, inputs, weights, archlen):
        activations = [inputs]
        zs = []  # Maybe add inputs for easier backpropagation

        for i in range(archlen - 2):
            z = np.dot(weights[i], activations[i])
            zs.append(z)
            activations.append(self.reLU(z))

        z = np.dot(weights[-1], activations[-1])
        zs.append(z)
        activations.append(self.softmax(z))
        return {"zs": zs,
                "activations": activations}


    def backpropagate(self, label, activations, zs, weights, LAYER_TYPE, new_weights):
        if type(new_weights) != list or len(new_weights) == 0:
            print("new_weights (gradient list) must be empty!")
            return

        if LAYER_TYPE == "FC":
            return self.backprop_fc(label, new_weights, weights, zs, activations, 0, len(activations))
        elif LAYER_TYPE == "CONV":
            pass

    #  gets first loss [aL - hot_vector] + new_weights + org_weights + zs + a[l]
    def backprop_fc(self, new_weights, weights, zs, acti, lr, er, layer):
        new_weights.append(lr * np.multiply(er, acti[layer - 1]))
        dzl = np.array(self.relu_prime(zs[layer - 1]))
        if layer > 0:
            er = np.multiply(np.dot(weights[layer].T, er), dzl)
            self.backprop_fc(new_weights, weights, zs, acti, lr, er, layer-1)
        # Need to flip the list of the gradients after backprop [new_weights]!
