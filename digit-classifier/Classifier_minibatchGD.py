import numpy as np
import matplotlib.pyplot as plt
import pickle


def normalize(arg):
    if type(arg) is not np.ndarray:
        arg = np.array(arg)
    return (arg - np.mean(arg)) / np.std(arg)


def displayMNIST(arr):
    arr = np.array(arr).reshape(28, 28)
    plt.imshow(arr, cmap="gray")
    plt.show()


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
    def __init__(self, arch=(), lr=0.1):
        self.arch = arch
        self.lr = lr
        self.loss = []
        self.accuracy_valid = []
        self.accuracy_test = []

        self.weights = []
        self.biases = []  # WIP

        self.zs = []
        self.activations = []

        for i in range(len(self.arch) - 1):
            self.weights.append(
                np.random.randn(arch[i + 1], arch[i]) * np.sqrt(2 / arch[i + 1])
            )

    def fit(self, inputs, labels, batchsize=500, epochs=100, iterations=1):
        # Fitting Done in Mini-Batch Gradient Descent.
        inputs = np.split(inputs[0: batchsize * epochs], epochs)
        labels = np.split(labels[0: batchsize * epochs], epochs)
        for _ in range(iterations):
            for i in range(len(labels)):
                loss = []
                weights = []
                input_batch = inputs[i]
                label_batch = labels[i]
                for k in range(len(self.weights)):
                    weights.append(np.zeros(self.weights[k].shape))
                for j in range(batchsize):
                    input = input_batch[j]
                    label = label_batch[j]
                    self.propagate(input)
                    # archive fix (2026): modern numpy no longer accepts ragged
                    # lists of arrays in np.add -- element-wise instead
                    weights = [w + dw for w, dw in zip(weights, self.back_prop(label))]
                    h = hot_vector(label, 10)
                    loss.append(
                        -np.sum(
                            h * np.log(self.activations[-1])
                            + (1 - h) * np.log(1 - self.activations[-1])
                        )
                    )
                    self.flush()

                self.loss.append(np.sum(loss) / batchsize)
                # archive fix (2026): same ragged-array issue as above
                self.weights = [w - dw / batchsize for w, dw in zip(self.weights, weights)]
                print("epoch {} :::: loss {}".format(i + 1, self.loss[-1]))

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
        new_weights = []
        er = self.activations[-1] - hot_vector(label, 10)
        for i in range(len(self.arch) - 2, -1, -1):
            dw = np.multiply(er.reshape(self.arch[i + 1], 1), self.activations[i])
            ap = self.lr * dw
            new_weights.append(ap)

            if i > 0:
                dzl = np.array(sigmoidp(self.zs[i - 1])).reshape(self.arch[i], 1)
                er = np.multiply(np.dot(self.weights[i].T, er).reshape(dzl.shape), dzl)

        return new_weights[::-1]

    def test(
            self, inputs, labels, epochs=5000, testing_method="validation", show_activations=False
    ):
        # testing_method = validation // testing
        totalac = 0
        c = 0
        for i in range(epochs):
            self.propagate(inputs[i])
            if np.max(self.activations[-1]) == self.activations[-1][labels[i][0]]:
                totalac += 1
            c += 1
            max_ = np.max(self.activations[-1])
            idx = [i for i, j in enumerate(self.activations[-1]) if j == max_]
            if c == 1:
                print("label [ ], network prediction [ ], digits      [    0,         1,         2,         3,          4,         5,         6,         7,         8,         9    ]",
                      end="\n")

            if show_activations:
                print("label {}, network prediction {}".format(labels[i], idx), end="")
                act = self.activations[-1] / float(np.sum(self.activations[-1]))
                act = ["{:.4%}".format(x) for x in act]
                print(", activations {}".format(act, end="\n"))
                displayMNIST(inputs[i])

            if testing_method == "validation":
                self.accuracy_valid.append(100 * totalac / c)
            elif testing_method == "testing":
                self.accuracy_test.append(100 * totalac / c)

            self.flush()

        print("Total Accuracy: {}".format(totalac / c * 100))

    def plot_loss(self):
        x = []
        [x.append(i + 1) for i in range(len(self.loss))]
        y = self.loss
        plt.xlabel = "epochs"
        plt.ylabel = "loss"
        plt.plot(x, y)
        plt.show()

    def plot_accuracy(self):
        x = []
        [x.append(i + 1) for i in range(len(self.accuracy_valid))]
        y = self.accuracy_valid

        x2 = []
        [x2.append(i + 1) for i in range(len(self.accuracy_test))]
        y2 = self.accuracy_test

        plt.xlabel = "Epochs"
        plt.ylabel = "Accuracy"
        plt.plot(x, y, "b")
        plt.plot(x2, y2, "g")
        plt.legend(["Validation Data", "Test Data"])
        plt.show()

    def export(self, name):
        NN_dict = {
            "arch": self.arch,
            "lr": self.lr,
            "loss": self.loss,
            "weights": self.weights,
            "accuracy_valid": self.accuracy_valid,
            "accuracy_test": self.accuracy_test,
        }
        with open(str(name + ".pickle"), "wb") as file:
            pickle.dump(NN_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    def import_nn(self, filename):
        with open(filename, "rb") as file:
            b = pickle.load(file)
        for element, value in b.items():
            setattr(self, element, value)

    def flush(self):
        self.activations = []
        self.zs = []
