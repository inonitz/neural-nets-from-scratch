from NN import NN
from mnist import MNIST
import time

if __name__ == "__main__":
    start = time.time()
    mndata = MNIST('samples', gz=True)

    Network = NN(784, 400, 2, 10)
    Network.set_training_type("stochastic")
    Network.set_learning_rate(0.001)
    Network.run(mndata, 0, 1000, 'train')
    # Network.run(mndata, 0, 20, 'test')
    # Network.type_run_arr = ['train', 'test']
    # Network.data_size_train = 15000
    print(time.time()-start)
    Network.graph_loss()
