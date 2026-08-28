import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import pickle


class Neural_Re:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.bias_weights = [0 for i in range(len(layers) - 1)]
        self.activated_sums = []
        self.learning_rate = .001

        self.error = []


        # Initialize Random Weights
        for i in range(len(layers) - 1):
            weights_matrix = np.multiply(np.random.rand(self.layers[i + 1], self.layers[i]), np.sqrt(1 / (self.layers[i] + self.layers[i + 1])))
            self.weights.append(weights_matrix)

    def add_bias(self, layer):
        # e.g: if layer == 0, layer in bias_weights is 0
        layer -= 1
        weights_matrix = np.random.rand(self.layers[layer + 1]).dot(
            np.sqrt(1 / (self.layers[layer] + self.layers[layer + 1])))
        self.bias_weights.insert(layer, weights_matrix)

    def set_input(self, inputs):
        inputs = np.array(inputs).reshape(self.layers[0], 1)
        self.activated_sums.append(inputs)

    @staticmethod
    def sigmoid(sums, derivative=False, activated=False):
        if activated:
            s = sums
        else:
            s = 1 / (1 + np.exp(-sums))
        if derivative:
            return s * (1 - s)
        else:
            return s

    @staticmethod
    def softmax(sums):
        exp = np.exp(sums)
        return exp / exp.sum()

    def train(self, data, labels, start=0, stop=0):
        if stop == 0:
            stop = len(labels)
        for i in range(start, stop):
            image = data[i]
            label = np.array([0 if labels[i] != j else 1 for j in range(10)]).reshape(10, 1)
            self.set_input(image)
            self.propagate()

            loss = self.calc_J(label, self.activated_sums[-1])
            self.backprop(label)
            self.error.append(loss)
            print("epoch: {}, loss {}".format(i+1, loss))

    def propagate(self):
        for layer in range(len(self.layers) - 1):
            zl = np.dot(self.weights[layer], self.activated_sums[layer])
            zl = np.add(zl, self.bias_weights[layer])

            if layer == len(self.weights) - 1:
                al = self.softmax(zl)
            else:
                al = self.sigmoid(zl)

            al = np.array(al).reshape(al.shape[0], 1)
            self.activated_sums.append(al)

    def backprop(self, target_vector):
        for layer in range(len(self.layers) - 1, 0, -1):
            error = np.subtract(self.activated_sums[layer], target_vector)
            gradient = self.sigmoid(self.activated_sums[layer], activated=True, derivative=True)
            gradient *= np.multiply(error, self.learning_rate)

            self.weights[layer - 1] -= np.dot(gradient, self.activated_sums[layer - 1].T)
            self.bias_weights[layer - 1] -= gradient

            t = np.dot(self.weights[layer - 1].T, error)
            t = np.multiply(t, self.sigmoid(self.activated_sums[layer - 1], activated=True, derivative=True))
            target_vector = t

        self.activated_sums = []

    # def SGD(self, layer):
    #     gradient_error = np.multiply(np.outer(self.derivative_sums[layer], self.activated_sums[layer]), self.learning_rate)
    #     self.weights[layer] = np.subtract(self.weights[layer], gradient_error)
    #     if self.bias_weights[layer] is not 0:
    #         gradient_bias_error = np.multiply(self.derivative_sums[layer], self.learning_rate)
    #         self.bias_weights[layer] = np.subtract(self.bias_weights[layer], gradient_bias_error)

    def calc_J(self, hot_vector, output):
        return -np.sum(np.multiply(hot_vector, np.log(output)))

    def graph_loss(self):
        x = self.error
        y = [i for i in range(len(self.error))]
        plt.plot(y, x, label="TRAINING PROCESS")
        plt.show()

    def debug(self, hot_vector):
        outputi = np.argmax(self.outputs)
        targeti = np.argmax(hot_vector)
        if outputi == targeti:
            self.success_count += 1

        accuracy = self.success_count / (len(self.error)) * 100
        print("Epoch: {}, loss: {}, accuracy: {}".format(len(self.error), self.loss_in_epoch, accuracy))









