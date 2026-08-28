import numpy as np
import matplotlib.pyplot as plt


def sigmoid(arg):
    return 1 / (1 + np.exp(-arg))


def sigmoidp(arg):
    return sigmoid(arg) * (1 - sigmoid(arg))


class XOR:
    def __init__(self, arch, lr=1, plot_loss=False):
        # neural network variables
        self.arch = arch
        self.lr = lr
        self.loss = []

        # arrays
        self.weights = []
        self.biases = [0 for i in range(len(arch))]
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
            z += self.biases[i]
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
            if type(self.biases[i]) is not int:
                temp = self.lr * err.reshape(self.biases[i].shape)
                self.biases[i] -= temp
            if i > 0:
                dz = np.array(sigmoidp(self.zs[i - 1])).reshape(self.arch[i], 1)
                el = np.multiply(self.weights[i].T * err, dz)
                err = el

    def add_bias(self, layer):
        if layer >= len(self.arch) - 1:
            print("Didn't Initialize bias! layer num doesn't exist / layer is output layer.")
        else:
            self.biases.insert(layer, np.random.randn(self.arch[layer+1]))
            print("initialized bias at layer {}(+1)".format(layer))

    def print_weights(self):
        [print("\n") for i in range(3)]
        for i in range(len(self.weights)):
            print("{} <-- weights, layer {}".format(self.weights[i], i))

        print("\nbiases:")
        for i in range(len(self.biases) - 1):
            print("{} <-- weights, layer {}".format(self.biases[i], i))

    def plot_loss(self):
        x = []
        [x.append(i + 1) for i in range(len(self.loss))]
        loss = self.loss
        plt.xlabel = "epochs"
        plt.ylabel = "loss"
        plt.plot(x, loss)
        plt.show()

    def flush(self):
        self.activations = []
        self.zs = []


ar = [2, 2, 1]
x = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
y = np.array([[1], [1], [0], [0]])
neural = XOR(arch=ar, lr=1, plot_loss=True)
neural.add_bias(1)
neural.fit(x, y, epochs=80000)
neural.test(x, y)
neural.print_weights()
neural.plot_loss()