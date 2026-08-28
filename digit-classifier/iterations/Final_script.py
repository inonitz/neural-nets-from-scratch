import numpy as np
import matplotlib.pyplot as plt


class Neural_Net:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.inputs = []
        self.plot_error = np.array([])
        self.learning_rate = .01

        for i in range(len(layers) - 1):
            weights_matrix = np.multiply(np.random.rand(self.layers[i + 1], self.layers[i]), np.sqrt(1 / (self.layers[i] + self.layers[i + 1])))
            self.weights.append(weights_matrix)

    def feed_forward(self, inputs):
        inputs = np.array(inputs).reshape(self.layers[0], 1)
        for i in range(len(self.layers) - 1):
            inputs = np.dot(self.weights[i], inputs)
            inputs = self.sigmoid(inputs)
        return inputs

    def epoch_training(self, epochs, data, labels, start=0, stop=0):
        stop = len(data)
        for i in range(epochs):
            for j in range(start, stop):
                sample = np.array(data[j])
                if type(labels[j]) is not int:
                    answer = np.array(labels[j])
                else:
                    answer = labels[j]
                self.train(sample, answer)
                MSE = self.mse(sample, answer)
                self.plot_error = np.append(self.plot_error, MSE)
                print("epoch {}, error: {}".format(i, MSE))

    def train(self, t_ex, label):
        inputs = np.array(t_ex).reshape(self.layers[0], 1)
        if type(label) is not int:
            targets = np.array(label).reshape(self.layers[-1], 1)
        else:
            targets = np.zeros(self.layers[-1])
            targets[label] = 1

        self.inputs = []
        self.inputs.append(inputs)
        for i in range(len(self.layers) - 1):
            inputs = np.dot(self.weights[i], inputs)
            inputs = self.sigmoid(inputs)
            self.inputs.append(inputs)

        # Calculating Error and updating weights:
        for i in range(len(self.layers) - 1, 0, -1):
            error = targets - self.inputs[i]
            gradient = self.sigmoid(self.inputs[i], activated=True, derivative=True)
            gradient *= np.multiply(error, self.learning_rate)
            print(gradient.shape, self.inputs[i - 1].T.shape)
            x = np.dot(gradient, self.inputs[i-1].T)
            print(x.shape)
            self.weights[i - 1] += x

            # Calculate Error of next Layer
            t = np.dot(self.weights[i - 1].T, error)
            # t += self.inputs[i - 1]
            targets = t

    def test(self, data, labels):
        for i in range(len(labels)):
            ex = data[i]
            label = labels[i]
            print("label: {}, prediction: {}, error: {}".format(label, self.feed_forward(ex), self.mse(ex, label)))

    def mse(self, inputs, target):
        t = np.array(target).reshape(self.layers[-1], 1)
        outputs = self.feed_forward(inputs)
        return np.sum((outputs - t) ** 2) / len(t)

    def plot_errors(self):
        y = [i+1 for i in range(len(self.plot_error))]
        plt.plot(y, self.plot_error, label="Gradient Descent Performance (Stochastic)")
        plt.show()

    @staticmethod
    def sigmoid(sums, derivative=False, activated=False):
        if not activated:
            s = 1 / (1 + np.exp(-sums))
        else:
            s = sums
        if derivative:
            return s * (1 - s)
        else:
            return s

