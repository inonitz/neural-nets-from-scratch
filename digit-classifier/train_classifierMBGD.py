import os
import matplotlib
from mnist.loader import MNIST
import numpy as np
from Classifier_minibatchGD import classifier
from Classifier_minibatchGD import normalize


mndata = MNIST("samples", gz=True)
ftraining_images, ftraining_labels = mndata.load_training()
testing_imgs, testing_lbls = mndata.load_testing()

training_images, training_labels = (
    normalize(ftraining_images[0:50000]),
    np.array(ftraining_labels[0:50000]).reshape(50000, 1),
)

validation_images, validation_labels = (
    normalize(ftraining_images[50000:60000]),
    np.array(ftraining_labels[50000:60000]).reshape(10000, 1),
)

testing_imgs, testing_lbls = (
    normalize(testing_imgs),
    np.array(testing_lbls).reshape(10000, 1),
)

if os.path.exists("NN.pickle"):
    a = classifier()
    a.import_nn("NN.pickle")
else:
    # no pretrained weights shipped in this archive -- train from scratch
    # (settings from the best run below: [784, 100, 100, 10], lr=1 --> ~97.39%)
    a = classifier([784, 100, 100, 10], lr=1)
    a.fit(training_images, training_labels, batchsize=400, epochs=125)
    a.export("NN")
# a.fit(training_images, training_labels, 400, 125, iterations=4)
# a.fit(training_images, training_labels, batchsize=400, epochs=125)

# Plot windows only make sense on an interactive backend; a headless run skips them
# (avoiding matplotlib's "FigureCanvasAgg is non-interactive" warning) -- use
# make_figures.py to render the figures to files instead.
_interactive = matplotlib.get_backend().lower() not in ("agg", "pdf", "ps", "svg", "template")

if _interactive:
    a.plot_loss()

# a.test(validation_images, validation_labels, epochs=10000, testing_method="validation", show_activations=True)
a.test(testing_imgs, testing_lbls, epochs=10000, testing_method="testing", show_activations=False)

if _interactive:
    a.plot_accuracy()
else:
    print("(non-interactive matplotlib backend -- skipping plot windows; run make_figures.py to save them)")
print("\n\n\n\n\n")

"""
How to adjust hyper-parameters:
1. Make sure training batches aren't too high, but aren't too low (don't burn the cpu)
2. adjust learning rate accordingly
"""

# batchsize=625, epochs=80 [784, 20, 20, 10], lr=5 --> accuracy ==> ~~93%
# batchsize=400, epochs=125 [784, 100, 10], lr=.1 --> accuracy ==> ~~93%
# batchsize=400, epochs=125, [784, 200, 10], lr=.1 --> accuracy ==> ~~95%
# batchsize=400, epochs=125, [784, 100, 100, 10], lr=1 --> accuracy ==> ~~96.5%
# batchsize=400, epochs=125, [784, 100, 100, 10], lr=1 --> accuracy ==> ~~97.39%
