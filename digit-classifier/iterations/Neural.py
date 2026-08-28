import numpy as np
import matplotlib.pyplot as plt


class NN:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.inputs = []
        self.sums = []
        self.errors = []
        self.learning_rate = 0.001

        self.errors_epochs = []
        self.errors_epoch = []
        for i in range(len(layers) - 1):
            weights_matrix = np.multiply(np.random.rand(self.layers[i], self.layers[i + 1]), np.sqrt(1 / (self.layers[i] + self.layers[i + 1])))
            self.weights.append(weights_matrix)

    @staticmethod
    def sigmoid(sums, derivative=False):
        for i in range(len(sums)):
            if sums[i] > 37:
                sums[i] = 37
            elif sums[i] < -37:
                sums[i] = -37

        s = 1 / (1 + np.exp(-sums))
        if derivative:
            return s * (1 - s)
        else:
            return s

    def tanh_dx(self, s):
        return 1 - np.tanh(s)**2

    def set_inputs(self, inputs):
        self.inputs.append(np.array(inputs))

    def propagate(self):
        for i in range(len(self.layers) - 1):
            zl = self.weights[i].T.dot(self.inputs[i])

            self.sums.append(zl)

            if i == len(self.weights) - 1:
                self.inputs.append(np.tanh(zl))
            else:
                self.inputs.append(self.sigmoid(zl))

        self.sums.insert(0, self.inputs[0])

    def backpropagate(self, vector):
        for i in range(len(self.layers) - 1, 0, -1):
            if i == len(self.layers) - 1:
                dJ = np.subtract(vector, self.inputs[-1])
                self.sums[i] = np.multiply(dJ, self.tanh_dx(self.sums[i]))

            else:
                sums_next_layer = np.sum(self.sums[i + 1].dot(self.weights[i].T))
                x = self.sigmoid(self.sums[i], derivative=True) + sums_next_layer
                self.sums[i] = x

        for i in range(len(self.weights)):
            self.sgd(i)

        self.errors_epoch.append((self.inputs[-1] - vector)**2)
        self.inputs = []
        self.sums = []

    def MSE(self, target, ex_amount):
        J = 1/ex_amount * np.sum(self.errors_epoch)
        self.errors_epochs.append(J)
        self.errors_epoch = []

    def sgd(self, i):
        i += 1
        gradient_error = self.sums[i]
        dJal = np.outer(self.inputs[i - 1], gradient_error)
        self.weights[i - 1] += np.multiply(self.learning_rate, dJal)

    def run(self, target, c, m):
        self.propagate()
        self.backpropagate(target)
        if c+1 == m:
            self.MSE(target, m)

    def plot(self):
        y = [i+1 for i in range(len(self.errors_epochs))]
        plt.plot(y, self.errors_epochs, label="Loss Over Epochs")
        plt.show()
