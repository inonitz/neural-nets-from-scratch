import numpy as np
from testdraft import Operate
import matplotlib.pyplot as plt
import pickle
from PIL import Image

"""
Use cifar_test.py (formerly ohaf.py) for reference (the way the file is "structured" and executed)

"""


def show_images(images, rows=None, cols=1, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        if rows is None:
            rows = np.ceil(n_images / float(cols))
        a = fig.add_subplot(cols, rows, n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        # a.set_title(title)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                        labelleft=False)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()





def unpickle(pf):
    with open(pf, 'rb') as upf:
        dicti = pickle.load(upf, encoding='bytes')
    return dicti


x = Operate(msg=False)
convnet_arch = {'[ [ Conv -> ReLU ] * 2 --> Pool ]*2 --> FC --> ReLU --> FC'}
# img = unpickle('data_batch_1')
img2 = Image.open("bird.jpg")
img2.load()
img2 = np.array(img2)


#  propagate over 1 layer only. [the layer will be customizable soon enough
#  with the Neural Net.py file finished.
# noinspection PyUnreachableCode
def propagate_convolution(image, filter, stride, filter_amount=16, pool=False, poolKS=(3, 3), pStride=2, pad=False):
    if type(filter) is tuple or type(filter) is list:
        filter = np.random.normal(0, np.sqrt(2. / image.shape[-1] * filter[0] * filter[1]),
                                  (filter[0], filter[1], filter_amount))
    if pad:
        image = x.pad3D(image, filter.shape[0], stride, -1)

    # zeros = np.zeros(x.comp)
    for i in range(filter_amount):
        convolved = np.array([x.convolve3D(image, filter[:, :, i], stride) for i in range(filter_amount)])
    print(convolved.shape)
    if not pool:
        return filter, x.reLU(convolved)

    return x.reLU(x.pooling3D(mat=convolved, filter_shape=poolKS, stride=pStride)), convolved, filter

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

            return convolved, reLU'd, pooled result.

    """


res, conv, fil = propagate_convolution(img2, (5, 5), 2, pool=True, poolKS=(2, 2), pStride=2, filter_amount=5, pad=False)
print("25% Done")
res2, conv2, fil2 = propagate_convolution(res, (3, 3), 2, pool=True, poolKS=(3, 3), pStride=2, filter_amount=5, pad=False)
print("50% Done.")
res3, conv3, fil3 = propagate_convolution(res2, (3, 3), 2, pool=True, poolKS=(2, 2), pStride=2, filter_amount=5, pad=False)
print("75% Done.")
res4, conv4, fil4 = propagate_convolution(res3, (3, 3), 2, pool=True, poolKS=(3, 3), pStride=2, filter_amount=128, pad=False)
print("Finished!")

print("First layer Result shapes: {} {} {}".format(res.shape, conv.shape, fil.shape))
print("Second layer Result shapes: {} {} {}".format(res2.shape, conv2.shape, fil2.shape))
print("Third layer Result shapes: {} {} {}".format(res3.shape, conv3.shape, fil3.shape))
print("Fourth layer Result shapes: {} {} {}".format(res4.shape, conv4.shape, fil4.shape))

l1, l2, l3 = [conv.T[i, :, :] for i in range(conv.shape[-1])], [fil[:, :, i] for i in range(fil.shape[-1])], [res[:, :, i] for i in range(res.shape[-1])]
l4, l5, l6 = [conv2.T[i, :, :] for i in range(conv2.shape[-1])], [fil2[:, :, i] for i in range(fil2.shape[-1])], [res2[:, :, i] for i in range(res2.shape[-1])]
l7, l8, l9 = [conv3.T[i, :, :] for i in range(conv3.shape[-1])], [fil3[:, :, i] for i in range(fil3.shape[-1])], [res3[:, :, i] for i in range(res3.shape[-1])]

# x.plot(img2, gray=True)
print(len(l1+l2+l3+l4+l5+l6+l7+l8+l9))
print(len(l3+l6+l9))

show_images(l1+l2+l3+l4+l5+l6+l7+l8+l9, 5, 9)

"""
show_images(l1+l2+l3+l4+l5+l6+l7+l8+l9, 21, 16)
use 28, 21 for filter_amount 16, 32, 64 [res-->res3]

res32, conv32, fil32 = propagate_convolution(img2, (3, 3), 2, pool=True, poolKS=(3, 3), poolS=2, filter_amount=32, pad=True)
show_images(res32, 32, 1)
"""

show_images(l3+l6+l9, 5, 3)

#  Remember Carefully that transposing of the tensor/matrix rotates it by 90 degrees (counter-clockwise)!!!
