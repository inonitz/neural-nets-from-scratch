from mnist.loader import MNIST
import numpy as np
from multilayer import Network
from Classifier_minibatchGD import normalize

mndata = MNIST("samples", gz=True)
training_imgs, training_lbls = mndata.load_training()
training_imgs, training_lbls = normalize(training_imgs),\
                              np.array(training_lbls)

testing_imgs, testing_lbls = mndata.load_testing()
testing_imgs, testing_lbls = normalize(testing_imgs),\
                              np.array(testing_lbls)


a = Network([784, 100, 100, 10], 1)
a.train(training_imgs[0:20000], training_lbls[0:20000], threshold=0.05)
a.plot_loss()
a.test(testing_imgs, testing_lbls, epochs=30)
