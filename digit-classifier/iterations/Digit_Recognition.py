import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.bias = np.zeros((len(self.layers) - 1))
        self.inputs = np.array([])

        self.learning_rate = 0.0001
        self.error = []

        for i in range(len(layers) - 1):
            weights_matrix = np.multiply(np.random.rand(self.layers[i + 1], self.layers[i]), np.sqrt(1 / (self.layers[i] + self.layers[i + 1])))
            self.weights.append(weights_matrix)

    def bias(self, layer):
        layer -= 1
        weights_matrix = np.multiply(np.random.rand(self.layers[layer + 1], self.layers[layer]),
                                     np.sqrt(1 / (self.layers[layer] + self.layers[layer + 1])))
        self.bias[layer] = weights_matrix


    @staticmethod
    def sigmoid(array, actv=False, deriv=False):
        if actv:
            s = array
        else:
            s = 1 / np.exp(-array)
        if deriv:
            return s * (1 - s)
        else:
            return s

    @staticmethod
    def softmax(array):
        e = np.exp(array)
        return np.divide(e, np.sum(e))

    def train(self, sample, label):
        # Reshape sample to Shape of inputs
        # Create 1 hot T vector
        sample = np.array(sample).reshape((self.layers[0], 1))

        # Propagate
        self.inputs = []
        self.inputs.append(sample)
        for i in range(len(self.layers) - 1):
            # W*X + b
            zL = np.matmul(self.weights[i], self.inputs[i])
            np.add(zL, self.bias[i], out=zL)
            if i == len(self.layers) - 2:
                aL = self.softmax(zL)
                self.inputs.append(aL)
            else:
                aL = self.sigmoid(zL)
                self.inputs.append(aL)

        # Backpropagate
        dJdz = np.array([])
        for i in range(len(self.layers) - 1, 0, -1):
            if i == len(self.layers) - 1:
                dJdz = np.subtract(self.inputs[-1], label) # Error at output layer
            else:
                dJda = np.dot(self.weights[i].T, dJdz) # Sum Corresponding weights * error of output layer
                dAdz = self.sigmoid(self.inputs[i], actv=True, deriv=True) # Derivation of Activation
                dJdz = dAdz * dJda

            # Gradient Descent
            aPL = self.inputs[i - 1].T  # activation in previous layer
            dJdw = np.dot(dJdz, aPL)
            self.weights[i - 1] -= self.learning_rate * dJdw

    def train_epoch(self, epochs, images, labels):
        for i in range(epochs):
            image = mndata.process_images_to_numpy(images[i])
            image = np.multiply(image, 1 / 256)
            label = labels[i]
            label = np.array([0 if label != j else 1 for j in range(self.layers[-1])]).reshape(
                self.layers[-1], 1)

            self.train(image, label)
            loss = -np.sum(label * np.log(self.inputs[-1]))
            print(i, "loss:", loss)
            self.error.append(loss)

    def plt(self):
        y = [i for i in range(len(self.error))]
        x = self.error
        plt.plot(y, x)
        plt.xlabel = "Epochs"
        plt.ylabel = "Loss"
        plt.show()


mndata = MNIST('samples', gz=True)
images, labels = mndata.load_training()
Digit = Classifier([784, 254, 10])
Digit.learning_rate = 0.001
Digit.train_epoch(20000, images, labels)
Digit.plt()
