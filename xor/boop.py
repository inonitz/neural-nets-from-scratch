import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidp(x):
    return sigmoid(x) * (1 - sigmoid(x))


x = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
y = np.array([[1], [1], [0], [0]])

arch = [2, 4, 1]
lr = 10
w1 = np.random.rand(arch[0], arch[1]).T
w2 = np.random.rand(arch[1], arch[2]).T
c = 0
loss = 0
for i in range(60000):
    inputs, label = x[c], y[c]
    z1 = np.dot(w1, inputs)
    a1 = sigmoid(z1)
    dz1 = np.array(sigmoidp(z1)).reshape(arch[1], 1)

    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)

    eL = np.dot(a2 - label, sigmoidp(z2))
    el = np.multiply(w2.T * eL, dz1)

    dw2 = np.multiply(eL, a1)
    dw1 = np.multiply(el, inputs)

    w2 -= lr * dw2
    w1 -= lr * dw1
    c += 1
    if c > 3:
        print("loss: {}".format((a2 - label)**2))
        c = 0
    if i > 59995:
        print(inputs, label, a2)

# print(w1)
# print(w2, '\n')
#
# print(a1, a2)
# print(dw2, "\n")
# print(dw1)
