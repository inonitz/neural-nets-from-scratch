from Neural_Refactored import Neural_Re
from mnist import MNIST


if __name__ == "__main__":
    mndata = MNIST('samples', gz=True)
    images, labels = mndata.load_training()

    NN = Neural_Re([784, 600, 600, 10])
    NN.learning_rate = .001
    NN.train(images, labels, 0, 2**12)

    NN.graph_loss()
