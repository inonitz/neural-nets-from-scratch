import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import pickle


class NN:
    def __init__(self, input, hidden, amount, output):
        self.input = input
        self.hidden = hidden
        self.amount = amount
        self.output = output

        self.learning_rate = 0.5
        self.errors = []  # Error through each Training Example
        self.errors_test = []
        self.error_signal = []  # The Gradient Error of each node in layer j

        self.inputs = []
        self.delta_sigmoid = []
        self.outputs = []
        self.weights = []
        self.layers = []
        self.nodes_in_between_every_layer = []
        self.initialize_NN()
        self.biases = [0] * len(self.nodes_in_between_every_layer)

        self.type_run_arr = []
        self.data_size_train = 0
        self.data_size_test = 0
        self.mini_size = 0  # Mini-batch
        self.epoch_count = 0  # General
        self.epoch_train = 0
        self.epoch_test = 0
        self.epoch_validate = 0
        self.training_type = "stochastic"
        self.type_run = "train"
        self.success_rate = 0

        # Add convenience list for different weight initializations
        # link for different initializations:
        # https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
        # Add convenience list for different activation functions
        # list: Tanh, Sigmoid, SoftArgMax, ReLU
        # [Make sure to have a derivative for each 1 {if available}]

        # Add different types of cost functions
        # link for different cost functions:
        # https://stats.stackexchange.com/questions/5363/backpropagation-algorithm?noredirect=1&lq=1
        # Add Convenience list for derivatives of the cost functions

        # Note for training procedure:
        # Add different ways of training the neural network:
        # Batch Gradient Descent, Mini-Batch Gradient Descent, Stochastic Gradient descent
        # Epoch = Finished Calculation of the Gradient in the type of training specified
        # Add Graphs for [accuracy, Epoch]

    def initialize_NN(self):
        global w
        self.layers.append(self.input)
        [self.layers.append(self.hidden) for i in range(self.amount)]
        self.layers.append(self.output)

        for i in range(len(self.layers) - 1):
            self.nodes_in_between_every_layer.append([])
            self.nodes_in_between_every_layer[i].append(self.layers[i])
            self.nodes_in_between_every_layer[i].append(self.layers[i + 1])

        # ----------------------------------> initialize weights randomly
        self.weights = []
        n = self.nodes_in_between_every_layer
        for i in range(len(n)):
            w = np.random.rand(n[i][1], n[i][0])
            # Gaussian initialization: mult by: 1 [or 2] / np.sqrt(n[i][0] + n[i][1])
            # He initialization: mult by: np.sqrt(2 [or 1] / n[i][0])
            np.multiply(w, np.sqrt(2 / (n[i][0] + n[i][1])), w)
            self.weights.append(w)
        # ----------------------------------> initialize empty input arrays for feed-forward
        for layer in self.layers:
            self.inputs.append(np.zeros(layer))
            self.delta_sigmoid.append(np.zeros(layer))
            self.error_signal.append(np.zeros(layer))
        self.delta_sigmoid.remove(self.delta_sigmoid[0])
        self.error_signal.remove(self.error_signal[0])

        # ----------------------------------> initialize empty Error Array for Back-Prop

    def add_bias(self, layer):
        n = self.nodes_in_between_every_layer
        try:
            x = np.random.rand(n[layer][1])
            self.biases[layer] = np.multiply(x, 1 / np.sqrt(n[layer][0] + n[layer][1]))
        except IndexError:
            print("Cant initialize to layer {:n} [Layer is Output layer///Not In Range]".format(layer))

    def set_inputs(self, inputs):
        if len(inputs) == len(self.inputs[0]):
            for i, inp in np.ndenumerate(inputs):
                self.inputs[0][i] = inp
        else:
            print("length of Inputs Given != length of Nodes In Input Layer")

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_training_type(self, type):
        if type == "mini-batch" or type == "stochastic" or type == "batch":
            self.training_type = type
            if type == "mini-batch":
                print("Please specify a size for each mini batch:")
                self.mini_size = int(input())
        else:
            print("{} is not a valid training type".format(type))

    def propagate(self):
        for idx, arr in np.ndenumerate(np.array(self.weights)):
            idx = idx[0]
            for i in range(len(arr)):
                sum = 0
                for j in range(len(self.inputs[idx])):
                    sum += self.inputs[idx][j] * arr[i][j]
                    if self.biases[idx] is not 0:
                        sum += self.biases[idx][i] * 1

                if idx == len(self.inputs) - 1:
                    self.inputs[idx + 1][i] = sum
                else:
                    if idx != len(self.delta_sigmoid) - 1:
                        self.delta_sigmoid[idx][i] = self.sigmoid(sum, derivative=True)
                    self.inputs[idx + 1][i] = self.sigmoid(sum, derivative=False)
        self.outputs = self.SoftMax(self.inputs[-1])

    def print_info(self):
        print("Neural Network Info:")
        print("Layers:", self.layers)
        print("Weights:", self.weights)
        print("Current Error:", self.errors, "\n", self.errors_test)
        print("training epochs:", self.epoch_train)
        print("testing epochs:", self.epoch_test)

    @staticmethod
    def map_np(array, val):
        r = np.zeros(array.shape)
        for i in range(len(array)):
            r[i] = array[i] * val
        return r

    @staticmethod
    def hot_vector(label, shape):
        vector = np.zeros(shape)
        vector[label] = 1
        return vector

    @staticmethod
    def tanh(sum, derivative=False):
        if derivative:
            return 1 - np.tanh(sum) ** 2
        else:
            return np.tanh(sum)

    @staticmethod
    def sigmoid(sum, derivative=False):
        # OVERFLOW CHECK
        if sum > 37:
            sum = 37
        elif sum < -37:
            sum = -37

        if derivative:
            return (1 / (1 + np.exp(-sum))) * (1 - (1 / (1 + np.exp(-sum))))
        else:
            return 1 / (1 + np.exp(-sum))

    @staticmethod
    def ReLU(sum):
        return max(0, sum)

    @staticmethod
    def SoftMax(output_vector):  # Take as input the (zj = wj * xj + b) and normalize it [0->1]
        sum = 0
        return_vector = np.copy(output_vector)
        for val in output_vector:
            sum += np.exp(val)
        for i, val in np.ndenumerate(output_vector):
            return_vector[i] = np.exp(val) / sum

        return return_vector

    def cross_entropy(self, target, gradient):
        if gradient is False:
            return -np.sum(target * np.log(self.outputs))

        elif gradient is True:
            self.error_signal[len(self.error_signal) - 1] = np.subtract(self.outputs, target)

    def backpropagate(self, target, flag):
        # Find Percentage of success over all examples trained / tested on
        flag_success = True
        for i in range(len(self.outputs)):
            if self.outputs[i] > self.outputs[target]:
                flag_success = False
        if flag_success is True:
            self.success_rate += 1

        target = self.hot_vector(target, 10)
        for layer in range(len(self.layers) - 1, -1, -1):
            if layer == len(self.layers) - 1:
                self.cross_entropy(target, gradient=True)
            else:
                for i in range(len(self.error_signal[layer - 1])):
                    error_sum = 0
                    for k in range(len(self.error_signal[layer])):
                        error_sum += self.error_signal[layer][k] * self.weights[layer][k][i]
                    self.error_signal[layer - 1][i] = self.delta_sigmoid[layer - 1][i] * error_sum

        if flag is True:
            for layer in range(len(self.weights)):
                for k in range(len(self.weights[layer])):
                    for j in range(len(self.weights[layer][k])):
                        gradient_w = self.error_signal[layer][k] * self.inputs[layer][j]

                        if self.biases[layer] is not 0:
                            self.biases[layer][k] -= self.learning_rate * self.error_signal[layer][k]
                        self.weights[layer][k][j] -= self.learning_rate * gradient_w

        if self.type_run == "train":
            self.errors.append(self.cross_entropy(target, gradient=False))
        elif self.type_run == "test":
            self.errors_test.append(self.cross_entropy(target, gradient=False))

    def run(self, data, start, end, type):
        print("{}ing".format(type))
        self.type_run = type
        self.type_run_arr.append(type)
        if type == "train":
            images, labels = data.load_training()
            self.data_size_train = end - start
        else:
            images, labels = data.load_testing()
            self.data_size_test = end - start

        if self.training_type == "stochastic":
            for i in range(start, end, 1):
                image = self.map_np(data.process_images_to_numpy(images[i]), 1 / 256)
                self.set_inputs(image)
                self.propagate()
                self.backpropagate(labels[i], True)
                self.epoch_count += 1
                temp = self.success_rate / self.epoch_count * 100
                print(self.epoch_count + start, "{}%".format(temp), self.errors[i])

        elif self.training_type == "mini-batch":
            cb = 0
            for i in range(start, end, 1):
                image = self.map_np(data.process_images_to_numpy(images[i]), 1 / 256)
                self.set_inputs(image)
                self.propagate()
                if cb == self.mini_size - 1:
                    self.backpropagate(labels[i], True)
                    cb = 0
                else:
                    self.backpropagate(labels[i], False)
                    cb += 1
                self.epoch_count += 1
                temp = self.success_rate / self.epoch_count * 100
                print(self.epoch_count + start, "{}%".format(temp))

        elif self.training_type == "batch":
            for i in range(start, end, 1):
                image = self.map_np(data.process_images_to_numpy(images[i]), 1 / 256)
                self.set_inputs(image)
                self.propagate()
                if self.epoch_count == end:
                    self.backpropagate(labels[i], True)
                else:
                    self.backpropagate(labels[i], False)
                self.epoch_count += 1
                temp = self.success_rate / self.epoch_count * 100
                print(self.epoch_count + start, "{}%".format(temp))

        if type == "train":
            self.epoch_train = self.epoch_count
        else:
            self.epoch_test = self.epoch_count
        self.epoch_count = 0
        self.success_rate = 0

    @staticmethod
    def plot(x, y, xt, yt, label, labelt):
        y = [i for i in range(y)]
        yt = [i for i in range(yt)]
        plt.title("Gradient Descent Performance for Loss Function")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(y, x, 'r', label=label)
        plt.plot(yt, xt, 'b', label=labelt)
        plt.show()

    def graph_loss(self):
        if self.training_type == "stochastic":
            x = self.errors
            y = self.epoch_train
            xt = self.errors_test
            yt = self.epoch_test
            self.plot(x, y, xt, yt, "Training", "Test")

        elif self.training_type == "mini-batch":
            x = []
            xt = []
            c = 0
            for type in self.type_run_arr:
                if type == 'train':
                    while c < self.data_size_train:
                        x.append(1 / self.mini_size * np.sum(self.errors[c: c + self.mini_size]))
                        c += self.mini_size
                elif type == 'test':
                    while c < self.data_size_test:
                        xt.append(1 / self.mini_size * np.sum(self.errors_test[c: c + self.mini_size]))
                        c += self.mini_size
                c = 0

            y = int(self.epoch_train / self.mini_size)
            yt = int(self.epoch_test / self.mini_size)
            self.plot(x, y, xt, yt, "Training", "Test")

    def export(self):
        NN_dict = {'learning_rate': self.learning_rate,
                   'errors': self.errors,
                   'errors_test': self.errors_test,
                   'weights': self.weights,
                   'biases': self.biases,
                   'epoch_train': self.epoch_train,
                   'epoch_test': self.epoch_test,
                   'epoch_validate': self.epoch_validate,
                   'data_size_train': self.data_size_train,
                   'data_size_test': self.data_size_test,
                   'type_run_arr': self.type_run_arr
                   }
        with open('NN.pickle', 'wb') as file:
            pickle.dump(NN_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    def Import(self, filename):
        with open(filename, 'rb') as file:
            b = pickle.load(file)
        for element, value in b.items():
            setattr(self, element, value)
