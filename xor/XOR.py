import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidp(x):
    return sigmoid(x) * (1 - sigmoid(x))


class XOR:
    def __init__(self, arch, lr=1, plot_loss=False):
        # neural network variables
        self.arch = arch
        self.lr = lr
        self.loss = []

        # arrays
        self.weights = []
        self.zs = []
        self.activations = []

        # flags
        self.plotloss = plot_loss

        for i in range(0, len(arch) - 1):
            self.weights.append(np.random.randn(arch[i + 1], arch[i]) * np.sqrt(2 / arch[i + 1]))
        # [print(self.weights[i], self.weights[i].shape) for i in range(len(arch) - 1)]

    def fit(self, inputs, labels, epochs=50000):
        c = 0
        for i in range(epochs):
            label = labels[c]
            input = inputs[c]
            self.propagate(input)
            self.back_prop(label)
            c += 1
            if c > 3:
                if self.plotloss:
                    self.loss.append((self.activations[-1] - label) ** 2)
                print("loss: {}".format((self.activations[-1] - label) ** 2))
                c = 0
            self.flush()

    def propagate(self, input):
        self.activations.append(input)
        for i in range(len(self.arch) - 1):
            z = np.dot(self.weights[i], self.activations[i])
            a = sigmoid(z)
            self.zs.append(z)
            self.activations.append(a)

    def test(self, inputs, labels):
        for i in range(len(inputs)):
            self.propagate(inputs[i])
            print(inputs[i], labels[i], self.activations[-1])
            self.flush()

    def back_prop(self, label):
        err = np.dot(self.activations[-1] - label, sigmoidp(self.zs[-1]))
        for i in range(len(self.arch) - 2, -1, -1):
            dw = np.multiply(err, self.activations[i])
            # print(dw)
            # print(dw.shape)
            self.weights[i] -= self.lr * dw

            if i > 0:
                dz = np.array(sigmoidp(self.zs[i - 1])).reshape(self.arch[i], 1)
                print(self.weights[i].T * err, dz.shape)
                el = np.multiply(self.weights[i].T * err, dz)
                err = el

    def plot_loss(self):
        x = []
        [x.append(i + 1) for i in range(len(self.loss))]
        y = self.loss
        plt.xlabel = "epochs"
        plt.ylabel = "loss"
        plt.plot(x, y)
        plt.show()

    def flush(self):
        self.activations = []
        self.zs = []


ar = [2, 3, 1]
x = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
y = np.array([[1], [1], [0], [0]])
neural = XOR(arch=ar, lr=1, plot_loss=True)
# neural.fit(x, y, epochs=60000)
# neural.test(x, y)
# neural.plot_loss()
neural.propagate(x[0])
neural.back_prop(y[0])
# [print(neural.activations[i].shape) for i in range(len(neural.activations))]
