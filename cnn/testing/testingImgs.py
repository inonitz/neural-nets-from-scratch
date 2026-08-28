from PIL import Image
from testing import Operations
import numpy as np
import matplotlib.pyplot as plt


def plot(mat):
    plt.imshow(mat)
    plt.show()


x = Operations.Operations()
img = Image.open('Screenshot_1.png')
img.load()

sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])


img = np.array([[3, 9, 5, 9],
                [1, 7, 4, 3],
                [2, 1, 6, 5]])


kernel = np.array([[0, 1, 0],
                   [0, 0, 0],
                   [0, -1, 0]])

print(x.conv2D(img, kernel, stride=2))
