import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidp(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(arg):
    return np.exp(arg) / np.sum(np.exp(arg))


def hot_vector(arg, vshape):
    if type(arg) is list:
        arg = arg[0]
    return np.array([0 if i != arg else 1 for i in range(vshape)])


class classifier(object):
    def __init__(self, arch, lr):
        self.arch = arch
        self.lr = lr
        self.loss = []

        self.weights = []
        self.zs = []
        self.activations = []

        for i in range(len(self.arch) - 1):
            self.weights.append(np.random.randn(arch[i + 1], arch[i]) * np.sqrt(2 / arch[i + 1]))

    def fit(self, inputs, labels, epochs=10000):
        for i in range(epochs):
            x = inputs[i]
            y = labels[i]
            h = hot_vector(y, 10)
            self.propagate(x)
            self.back_prop(y)
            loss = -np.sum(h*np.log(self.activations[-1]) + (1-h) * np.log(1-self.activations[-1]))
            print('epoch: {}, loss: {}'.format(i + 1, loss))

            self.loss.append(loss)
            self.flush()
            # start with SGD , then add mini batch, batch

    def propagate(self, input):
        self.activations.append(input)
        for i in range(len(self.arch) - 1):
            z = np.dot(self.weights[i], self.activations[i])
            if i == len(self.arch) - 2:
                acti = softmax(z)
            else:
                acti = sigmoid(z)
            self.zs.append(z)
            self.activations.append(acti)

    def back_prop(self, label):
        er = self.activations[-1] - hot_vector(label, 10)
        for i in range(len(self.arch) - 2, -1, -1):
            dw = np.multiply(er.reshape(self.arch[i + 1], 1), self.activations[i])
            # print(dw)
            # print(dw.shape)
            self.weights[i] -= self.lr * dw

            if i > 0:
                dzl = np.array(sigmoidp(self.zs[i - 1])).reshape(self.arch[i], 1)
                er = np.multiply(np.dot(self.weights[i].T, er).reshape(dzl.shape), dzl)
                # print(er.shape)

    def test(self, inputs, labels, epochs=10):
        totalac = 0
        c = 0
        for i in range(epochs):
            self.propagate(inputs[i])
            if np.max(self.activations[-1]) == self.activations[-1][labels[i][0]]:
                totalac += 1
            c += 1
            print('label: {} ::::::: accuracy {}% :::::: Total accuracy:   {}%'.format(labels[i], self.activations[-1][labels[i]]*100, totalac/c*100))

            # h = hot_vector(labels[i], 10)
            # self.loss.append(-np.sum(h*np.log(self.activations[-1]) + (1-h) * np.log(1-self.activations[-1])))
            self.flush()

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

    def flush_loss(self):
        self.loss = []


