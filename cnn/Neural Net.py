import numpy as np
from operate import operations


def in_args(key, obj_type, kwargs):
    if key in kwargs:
        if type(kwargs.get(key)) == obj_type:
            return kwargs.get(key)


def repc(x):
    a = [x.replace(c, '') for c in x if 48 <= ord(c) <= 57]
    if not a: return x
    else: return a[-1]


# self.net_arch = in_args('arch', dict, kwargs)
# self.conv_args = in_args('conv', dict, kwargs)
# self.pool_args = in_args('pool', dict, kwargs)
# self.FC_args = in_args('FC', dict, kwargs)


class Network:
    def __init__(self):
        self.Op = operations()
        self.layers = {}

    """
    Add Function to add Concurrent (used) Layers. [Based on args, conv_arch]
    Don't Over-Engineer Until everything is working! Don't get lazy when you're finished!
    """

    def ADD_LAYER(self, **kwargs):
        for key in kwargs:
            temp = repc(key)
            if temp.lower() == 'conv' or temp.lower() == 'pool' or temp.lower() == 'fc':
                self.CREATE_LAYER(key, kwargs.get(key))

    def CREATE_LAYER(self, name, name_args):
        if type(name_args) != dict and type(name_args) != list:
            raise Exception("Function CREATE_LAYER Only accepts dictionaries // lists as viable layer arguments.")

        if name not in self.layers:
            self.layers[name] = name_args
        else:
            self.layers[name] = name_args

        if 'layers' in name_args.keys():
            a = name_args['layers']
            name_args['weights'] = [np.random.rand(a[i+1], a[i]) * np.sqrt(2/a[i+1]) for i in range(len(a)-1)]

    # noinspection PyArgumentList
    def printSCHEMA(self):
        for key, value in self.layers.items():
            print("===========================||\nLayer: {} \nParameters:".format(key))
            for k, v in value.items():
                if type(v) == np.ndarray:
                    print("{}: {} shape: {}".format(k, type(v), v.shape))
                elif k == 'weights':
                    [print("layer {}: {}".format(i+1, v[i].shape)) for i in range(len(v))]
                else:
                    print(k, ":", v)



x = Network()
img = np.random.rand(32, 32)
stride = 2
filter_size = (3, 3)
x.ADD_LAYER(conv={'img': img,
                  'stride': stride,
                  'filter': filter_size},
            pool={'stride': 2,
                  'filter': (2, 2)},
            conv1={},
            pool1={},
            FC={'layers': [4, 3, 2],
                'lr': 1e-5})

x.printSCHEMA()
