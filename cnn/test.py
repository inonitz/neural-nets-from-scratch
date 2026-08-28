import numpy as np
from operate import operations

x = np.random.rand(50, 50, 3)
a = operations()

l = [4, 3, 2]
new = []
inputs = np.random.rand(4, )
weights = [np.random.rand(3, 4), np.random.rand(2, 3)]
ac = a.feedforward(inputs, weights, len(l))
acti = ac["activations"]
zs = ac["zs"]
lr = 0.5
er = acti[-1] - a.hot_vector(0, 2)

[print(zs[i].shape) for i in range(len(zs))]
a.backprop_fc(new, weights, zs, acti, lr, er, len(l))
