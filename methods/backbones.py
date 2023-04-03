import utils
import torch.nn as nn


class NNEncoder(nn.Module):
    def __init__(self, args, view_shape=[1, 2, 3, 4], v=0):
        super().__init__()
        channels_list = []
        if args.channels.count(-1) == 1:
            channels_list.append(args.channels.copy())
            v = 0
        else:
            channels_list = utils.deal(args.channels.copy(), -1)
        channels = channels_list[v]
        channels[0] = view_shape[-1]
        self.blocks = nn.Sequential()
        for i in range(1, len(channels)):
            self.blocks.add_module('nn%d' % i, nn.Linear(channels[i-1], channels[i]))
            self.blocks.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.blocks(x)


class NNDecoder(nn.Module):
    def __init__(self, args, view_shape=[1, 2, 3, 4], v=0):
        super().__init__()
        channels_list = []
        if args.channels.count(-1) == 1:
            channels_list.append(args.channels.copy())
            v = 0
        else:
            channels_list = utils.deal(args.channels.copy(), -1)
        channels = channels_list[v]
        channels[0] = view_shape[-1]
        channels = list(reversed(channels))
        self.blocks = nn.Sequential()
        for i in range(1, len(channels)):
            self.blocks.add_module('nn%d' % i, nn.Linear(channels[i-1], channels[i]))
            self.blocks.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.blocks(x)
