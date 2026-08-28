import numpy as np


class NeuralClassifier:
    def __init__(self, layers):
        self.layers = layers

        self.weights = []
        self.inputs = []
        self.bias = [0 for i in range(len(self.layers) - 1)]

        self.error_list = []
        self.learning_rate = .1

        for i in range(len(layers) - 1):
            weights = np.multiply(np.random.rand(layers[i], layers[i + 1]), 1 / (layers[i + 1] + layers[i]))
            self.weights.append(weights)


    @staticmethod
    def sigmoid(x):
        return np.array(1 / np.exp(x))

    @staticmethod
    def sigmoid_p(x):
        return np.matmul(x, 1 - x)

    @staticmethod
    def softmax(x):
        expA = np.exp(x)
        return expA / np.sum(expA)

    def feed_forward(self, inputs):
        inputs = np.array(inputs).reshape(len(inputs), -1)
        self.inputs.append(np.array(inputs))

        for i in range(0, len(self.weights)):

            zL = np.dot(self.weights[i].T, self.inputs[i]) + self.bias[i]
            aL = self.sigmoid(zL)
            self.inputs.append(aL.reshape(aL.shape[0], 1))

    def backpropagate(self, target):
        dJdz = np.array([])
        for i in range(len(self.layers) - 1, 0, -1):
            if i == len(self.layers) - 1:
                dJdz = (self.inputs[-1] - target)
                self.error_list.append(dJdz)
                dJdz = np.matmul(dJdz, self.sigmoid_p(self.inputs[-1]))
            else:
                dJdz = np.sum(np.dot(self.weights[i], dJdz))
                dJdz = np.dot(self.sigmoid_p(self.inputs[-1]), dJdz)

            updated_weights = self.learning_rate / (len(self.error_list)) * np.dot(self.inputs[i - 1], dJdz)
            updated_bias_weights = np.multiply(self.learning_rate / (len(self.error_list)), dJdz)
            self.weights[i - 1] += updated_weights
            self.bias[i - 1] += updated_bias_weights

    def fit(self, epochs, training_data, answer_data):
        for i in range(epochs):
            for j in range(len(answer_data)):
                answer = np.array(answer_data[j]).reshape(self.layers[-1], -1)
                sample = np.array(training_data[j]).reshape(self.layers[0], -1)
                self.feed_forward(sample)
                self.backpropagate(answer)

            print(i, "loss:", 1/len(answer_data) * np.sum([val**2 for val in self.error_list]))

    def test(self, training_data, answer_data):
        for i in range(len(answer_data)):
            print(i)
            answer = np.array(answer_data[i]).reshape(self.layers[-1], -1)
            sample = np.array(training_data[i]).reshape(self.layers[0], -1)
            self.feed_forward(sample)
            print("predicted: {}, target: {}, loss: {}".format(self.inputs[-1], answer, 1/len(answer_data) * np.sum([val**2 for val in self.error_list])))


if __name__ == "__main__":
    inputs = [[0, 1], [1, 1], [0, 0], [1, 0]]
    ans = [1, 0, 0, 1]
    x = NeuralClassifier([2, 3, 1])
    x.fit(5000, training_data=inputs, answer_data=ans)
    x.test(inputs, ans)
