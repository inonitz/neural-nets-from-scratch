from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
import random
import time


def SoftMax(output_vector):  # Take as input the (zj = wj * xj + b) and normalize it [0->1]
    sum = 0
    for val in output_vector:
        sum += np.exp(val)
    for i, val in np.ndenumerate(output_vector):
        output_vector[i] = np.exp(val) / sum

    return output_vector


def Sigmoid(sum, derivative):
    if derivative is False:
        return 1 / (1 + np.exp(-sum))
    else:
        return (1 / (1 + np.exp(-sum))) * (1 - (1 / (1 + np.exp(-sum))))


def map_np(array, val):
    r = np.zeros(array.shape)
    for i in range(len(array)):
        r[i] = array[i] * val
    return r


def cross_entropy(target, gradient):
    if gradient is False:
        r = -np.sum(target * np.log(outputs))
        errors.append(r)
    elif gradient is True:
        Errors[len(Errors) - 1] = np.subtract(outputs, target)


def add_bias(layer):
    try:
        x = np.random.rand(y[layer][1])
        biases[layer] = np.multiply(x, 1 / np.sqrt(y[layer][0] + y[layer][1]))
    except IndexError:
        print("LAYER ISN'T IN RANGE OR IS OUTPUT LAYER.", layer)


def init_layers(layer_list, hidden_nodes=5, hidden_nodes_layers=2, output_nodes=10):
    ret_list = []
    if layer_list is None:
        layer_list = [10]
    [layer_list.insert(layer + 1, hidden_nodes) for layer in range(hidden_nodes_layers)]
    layer_list.insert(len(layer_list), output_nodes)

    for i in range(len(layer_list) - 1):
        ret_list.append([])
        ret_list[i].append(layer_list[i])
        ret_list[i].append(layer_list[i + 1])

    return ret_list


def init_weights(layers):
    weights = []
    for i in range(len(layers)):
        weights.append([[]] * layers[i][0])
        for j in range(layers[i][1]):
            weights[i][j].append(0)

    pprint(weights)
    return weights


def weights_np(layers):
    weights = []
    for i in range(len(layers)):
        w = np.random.rand(layers[i][1], layers[i][0])
        np.multiply(w, 1 / np.sqrt(layers[i][0] + layers[i][1]), w)
        weights.append(w)

    # print(weights, np.array(weights))
    return np.array(weights)


start = time.time()
if __name__ == "__main__":
    x = init_layers(None)
    x = weights_np(x)
    # print(x)
    # for iterate in range(1):
    #     for i in np.nditer(x, flags=["refs_ok"]):
    #         print(i)
    #         print("-------------")
# [print("-------------") for i in range(3)]

inputs = np.array([0, 0.5, 0.1, 0.89513, 0.13783, 0.51538, 0.513, 0.9814, 0.6235, 1])
inputs_layer2 = np.zeros(5, dtype=np.float64)
inputs_layer3 = np.zeros(5, dtype=np.float64)
inputs_layer4 = np.zeros(10, dtype=np.float64)
total_inputs = [inputs, inputs_layer2, inputs_layer3, inputs_layer4]
print("All inputs:", total_inputs)


# testing bias array: \\\\
y = init_layers(None)
# l = [y[i][0] for i in range(len(y))]
biases = [0] * len(y)
# for idx in range(len(l)):
#     print(y[idx][1])
#     biases.append(np.random.rand(y[idx][1]))
#
#
# print(biases)

# add_bias(1)
# print(biases)

# [print("--------") for i in range(3)]

add_bias(1)
add_bias(2)
layers = [10, 5, 5, 10]
errors = []
derivatives_nodes = []
Errors = []
for layer in layers:
    Errors.append(np.zeros(layer))
    derivatives_nodes.append(np.zeros(layer))

for ix, arrayw in np.ndenumerate(x):
    ix = ix[0]
    # print(total_inputs[ix+1])
    # print(ix)
    for i in range(len(arrayw)):
        sum = 0
        for j in range(len(total_inputs[ix])):
            sum += total_inputs[ix][j] * arrayw[i][j]
            if biases[ix] is not 0:
                sum += biases[ix][i] * 1
            # print(arrayw[i][j], total_inputs[ix][j])

        if ix == len(x)-1:
            total_inputs[ix+1][i] = sum
        else:
            derivatives_nodes[ix + 1][i] = Sigmoid(sum, derivative=True)
            total_inputs[ix+1][i] = Sigmoid(sum, derivative=False)

total_inputs[len(total_inputs)-1] = SoftMax(total_inputs[len(total_inputs)-1])
print("printing Output array, weights for validation:")
print("Current Weights: ", x)
print("-----------------------------------")
print("Propagation through the network:", total_inputs)
print("-----------------------------------")
print("Biases:", biases)
print("-----------------------------------")
# !!!!!!!!!! Make sure input array sizes are the same as the amount of nodes
# Besides output = 2 nodes that doesnt work

# test = total_inputs[len(total_inputs)-1]
# test = SoftMax(test)
# print(test)

# s = 0
# a = np.array([0.30888617, 0.24382146, 0.20903463, 0.23825774])
# for x in a:
#     s += x
#
# print(s)


# mndata = MNIST('samples', gz=True)
# images, labels = mndata.load_training()
# i = random.randrange(0, len(images))
# print(mndata.display(images[i]))
# n = mndata.process_images_to_numpy(images[i])
# n.dtype = np.int32
# print(n)
#
# z = map_np(n, 1 / 256)
# print(z.reshape(28, 28))


def displayMNIST(arr):
    arr = arr.reshape(28, 28)
    plt.imshow(arr, cmap='gray')
    plt.show()


# displayMNIST(z)
# displayMNIST(n)


def hot_vector(label, shape):
    vector = np.zeros(shape)
    vector[label] = 1
    return vector


# a = np.array([0.09999478, 0.10000278, 0.10000341, 0.10000204, 0.09999278,
#               0.1000069, 0.10000687, 0.09999682, 0.09999514, 0.09999849])
# print(SoftMax(a, derivative=False))


print(biases)
outputs = total_inputs[len(total_inputs)-1]
target = hot_vector(9, 10)
Errors.remove(Errors[0])
derivatives_nodes.remove(derivatives_nodes[0])
print("f'(zjl):", derivatives_nodes)
for layer in range(len(layers) - 1, 0, -1):
    print("Layer:", layer)
    if layer == len(layers) - 1:
        cross_entropy(target, gradient=True)  # Compute Error signal for all nodes in Output layer
    else:
        for i in range(len(Errors[layer - 1])):
            error_sum = 0  # Calculate Sum of Errors in Above Layer * corresponding weights
            for k in range(len(Errors[layer])):
                # print("layer:", layer, "i", i, "k", k)
                c = Errors[layer][k] * x[layer][k][i]
                error_sum += c
            Errors[layer - 1][i] = derivatives_nodes[layer - 1][i] * error_sum
        print(Errors[layer - 1])

print("---------------------------")
print("Gradient Error Signal Of each node in each layer:", Errors)
lr = 1
print(biases)
for layer in range(len(x)):
    # print("Weights", x[layer])
    # print("Error signal for each neuron in layers", Errors[layer])
    # print("Inputs", total_inputs[layer])
    # print("----------------------------")
    for k in range(len(x[layer])):
        for j in range(len(x[layer][k])):
            gradient_w = x[layer][k][j]*Errors[layer][k]*total_inputs[layer][j]
            if biases[layer] is not 0:
                biases[layer][k] -= lr*Errors[layer][k]
            x[layer][k][j] -= lr*gradient_w
            # print("{}, {}".format(k, j), x[layer][k][j], "*", Errors[layer][k], "*", total_inputs[layer][j], " Result:", result)
    # print("---------------")

errors.append(cross_entropy(target, gradient=False))
print("Updated Weights:", x)
print(biases)
print(errors)

print(time.time() - start)

x = [[8.72375478e+13],
 [9.05239204e+13],
 [9.65167993e+13],
 [1.01260624e+14],
 [9.77055659e+13],
 [9.02011433e+13],
 [1.07631627e+14],
 [9.85212254e+13],
 [1.07367063e+14],
 [9.96001080e+13]]


