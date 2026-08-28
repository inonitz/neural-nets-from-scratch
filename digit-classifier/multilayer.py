import numpy as np
import Classifier_minibatchGD as cm  # archive fix (2026): flat layout, no 'project' package
import matplotlib.pyplot as plt


def hot_vector(arg, vshape):
    if type(arg) is list:
        arg = arg[0]
    return np.array([0 if i != arg else 1 for i in range(vshape)])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidp(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(arg):
    return np.exp(arg) / np.sum(np.exp(arg))


class Network:
    def __init__(self, arch, lr=.1):
        self.arch = arch
        self.lr = lr
        self.loss = []

        self.weights = []
        self.zs = []
        self.activations = []

        for i in range(len(self.arch) - 1):
            self.weights.append(np.random.randn(arch[i + 1], arch[i]) * np.sqrt(2 / arch[i + 1]))

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

    def backpropagate(self, label):
        new_weights = []
        er = self.activations[-1] - hot_vector(label, 10)
        for i in range(len(self.arch) - 2, -1, -1):
            dw = np.multiply(er.reshape(self.arch[i + 1], 1), self.activations[i])
            ap = self.lr * dw
            new_weights.append(ap)

            if i > 0:
                dzl = np.array(sigmoidp(self.zs[i - 1])).reshape(self.arch[i], 1)
                er = np.multiply(np.dot(self.weights[i].T, er).reshape(dzl.shape), dzl)

        # archive fix (2026): modern numpy rejects ragged lists of arrays -- element-wise instead
        self.weights = [w - dw for w, dw in zip(self.weights, new_weights[::-1])]

    def train(self, inputs, labels, threshold=.1):
        temp = 1000
        while temp > threshold:
            for i in range(len(labels)):
                input = inputs[i]
                label = labels[i]

                self.propagate(input)
                self.backpropagate(label)
                h = hot_vector(label, 10)
                self.loss.append(-np.sum(h * np.log(self.activations[-1]) + (1 - h) * np.log(1 - self.activations[-1])))
                temp += self.loss[-1]
                print("epoch: {}, loss: {}".format(i, self.loss[-1]))
                self.flush()
            temp /= len(labels)
            print(temp)

    def test(self, inputs, labels, epochs):
        for i in range(epochs):
            self.propagate(inputs[i])
            cm.displayMNIST(inputs[i])
            print("label {}, prediction {}".format(labels[i], self.activations[-1][labels[i]]))
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


