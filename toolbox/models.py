import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Callable
from collections import namedtuple
from copy import deepcopy


class Hyperparameters:
    def __init__(self, in_channels: int=None, hidden_channels: int=None, out_channels: int=None, dropout_rate: float=None):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

class Container(nn.Module):
    def __init__(self, hp: Hyperparameters, model: nn.Module):
        super().__init__()
        self.hp = hp
        self.out_channels = self.hp.out_channels
        self.adaptor = nn.Conv2d(self.hp.in_channels, self.hp.hidden_channels, 1, 1)
        self.BN_adapt = nn.BatchNorm2d(self.hp.hidden_channels)
        self.model = model(self.hp)
        self.compress = nn.Conv2d(self.hp.hidden_channels, self.hp.out_channels, 1, 1)
        self.BN_out = nn.BatchNorm2d(self.hp.out_channels)

    def forward(self, x):
        x = self.adaptor(x)
        x = self.BN_adapt(F.relu(x, inplace=True))
        z = self.model(x)
        z = self.compress(z)
        z = self.BN_out(F.relu(z, inplace=True)) # need to try without to see if it messes up average gray level
        return z


class HyperparametersUnet(Hyperparameters):
    def __init__(self, nf_table: List[int], kernel_table: List[int], stride_table: List[int], pool:bool, **kwargs):
        super().__init__(**kwargs)
        self.hidden_channels = nf_table[0]
        self.nf_table = nf_table
        self.kernel_table = kernel_table
        self.stride_table = stride_table
        self.pool = pool


class Unet(nn.Module):
    def __init__(self, hp: HyperparametersUnet):
        super().__init__()
        self.hp = deepcopy(hp) # pop() will modify lists in place
        self.nf_input = self.hp.nf_table[0]
        self.nf_output = self.hp.nf_table[1]
        self.hp.nf_table.pop(0)
        self.kernel = self.hp.kernel_table.pop(0)
        self.stride = self.hp.stride_table.pop(0)
        self.dropout_rate = self.hp.dropout_rate

        self.dropout = nn.Dropout(self.dropout_rate)
        self.conv_down = nn.Conv2d(self.nf_input, self.nf_output, self.kernel, self.stride)
        self.BN_down = nn.BatchNorm2d(self.nf_output)
        self.conv_up = nn.ConvTranspose2d(self.nf_output, self.nf_input, self.kernel, self.stride)
        self.BN_up = nn.BatchNorm2d(self.nf_input)

        if len(self.hp.nf_table) > 1:
            self.unet2 = Unet(self.hp)
        else:
            self.unet2 = None

        self.reduce = nn.Conv2d(2*self.nf_input, self.nf_input, 1, 1)
        self.BN_out = nn.BatchNorm2d(self.nf_input) # optional ?

    def forward(self, x):

        y = self.dropout(x)
        y = self.conv_down(y)
        y = F.relu(self.BN_down(y)) # y = self.BN_down(F.relu(y, inplace=True))

        if self.unet2 is not None:
            if self.hp.pool:
                y_size = y.size()
                y, indices = F.max_pool2d(y, 2, stride=2, return_indices=True)
            y = self.unet2(y)
            if self.hp.pool:
                y = F.max_unpool2d(y, indices, 2, stride=2, output_size=y_size)

        y = self.dropout(y)
        y = self.conv_up(y)
        y = self.BN_up(F.relu(y, inplace=True))
        y = torch.cat((x, y), 1)
        y = self.reduce(y)
        y = self.BN_out(F.relu(y, inplace=True))
        return y


class HyperparametersCatStack(Hyperparameters):
    def __init__(self, N_layers, kernel, padding, stride, **kwargs):
        super().__init__(**kwargs)
        self.N_layers = N_layers
        self.kernel = kernel
        self.padding = padding
        self.stride = stride


class ConvBlock(nn.Module):
    def __init__(self, hp: Hyperparameters):
        self.hp = hp
        super().__init__()
        self.dropout = nn.Dropout(self.hp.dropout_rate)
        self.conv = nn.Conv2d(self.hp.hidden_channels, self.hp.hidden_channels, self.hp.kernel, self.hp.stride, self.hp.padding)
        self.BN = nn.BatchNorm2d(self.hp.hidden_channels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.BN(F.relu(x, inplace=True))
        return x


class CatStack(nn.Module):

    def __init__(self, hp: Hyperparameters):
        super().__init__()
        self.hp = hp
        self.conv_stack = nn.ModuleList()
        for i in range(self.hp.N_layers):
            self.conv_stack.append(ConvBlock(hp))
        self.reduce = nn.Conv2d((1 + self.hp.N_layers) * self.hp.hidden_channels, self.hp.hidden_channels, 1, 1)
        self.BN = nn.BatchNorm2d(self.hp.hidden_channels)

    def forward(self, x):
        x_list = [x]
        for i in range(self.hp.N_layers):
            x = self.conv_stack[i](x)
            x_list.append(x)
        x = torch.cat(x_list, 1)
        x = self.reduce(x)
        y = self.BN(F.relu(x, inplace=True))
        return y
