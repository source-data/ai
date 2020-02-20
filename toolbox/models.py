import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Callable, ClassVar
from collections import namedtuple
from copy import deepcopy


class Hyperparameters:
    def __init__(self, in_channels: int=None, hidden_channels: int=None, out_channels: int=None, dropout_rate: float=None):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

    def __str__(self):
        return "; ".join([f"{a}={v}" for a, v in self.__dict__.items()])


class Container(nn.Module):
    def __init__(self, hp: Hyperparameters, model: ClassVar, conv: ClassVar, bn: ClassVar):
        super().__init__()
        self.hp = hp
        self.out_channels = self.hp.out_channels
        self.adaptor = conv(self.hp.in_channels, self.hp.hidden_channels, 1, 1)
        self.BN_adapt = bn(self.hp.hidden_channels)
        self.model = model(self.hp)
        self.compress = conv(self.hp.hidden_channels, self.hp.out_channels, 1, 1)
        self.BN_out = bn(self.hp.out_channels)

    def forward(self, x):
        x = self.adaptor(x)
        x = self.BN_adapt(F.relu(x, inplace=True))
        z = self.model(x)
        z = self.compress(z)
        z = self.BN_out(F.relu(z, inplace=True)) # need to try without to see if it messes up average gray level
        return z


class Container1d(Container):
    def __init__(self, hp: Hyperparameters, model: ClassVar):
        super().__init__(hp, model, nn.Conv1d,  nn.BatchNorm1d)


class Container2d(Container):
    def __init__(self, hp: Hyperparameters, model: ClassVar):
        super().__init__(hp, model, nn.Conv2d,  nn.BatchNorm2d)


class HyperparametersUnet(Hyperparameters):
    def __init__(self, nf_table: List[int], kernel_table: List[int], stride_table: List[int], pool:bool, **kwargs):
        super().__init__(**kwargs)
        self.hidden_channels = nf_table[0]
        self.nf_table = nf_table
        self.kernel_table = kernel_table
        self.stride_table = stride_table
        self.pool = pool


class Unet(nn.Module):
    def __init__(self, hp: HyperparametersUnet, conv: ClassVar, convT: ClassVar, bn: ClassVar, pool: Callable, unpool: Callable):
        super().__init__()
        self.hp = deepcopy(hp) # pop() will modify lists in place
        self.nf_input = self.hp.nf_table[0]
        self.nf_output = self.hp.nf_table[1]
        self.hp.nf_table.pop(0)
        self.kernel = self.hp.kernel_table.pop(0)
        self.stride = self.hp.stride_table.pop(0)
        self.dropout_rate = self.hp.dropout_rate

        self.dropout = nn.Dropout(self.dropout_rate)
        self.conv_down = conv(self.nf_input, self.nf_output, self.kernel, self.stride)
        self.BN_down = bn(self.nf_output)
        self.pool = pool
        self.conv_up = convT(self.nf_output, self.nf_input, self.kernel, self.stride)
        self.unpool = unpool
        self.BN_up = bn(self.nf_input)

        if len(self.hp.nf_table) > 1:
            self.unet = self.__class__(self.hp)
        else:
            self.unet = None

        self.reduce = conv(2*self.nf_input, self.nf_input, 1, 1)
        self.BN_out = bn(self.nf_input)


    def forward(self, x):

        y = self.dropout(x)
        y = self.conv_down(y)
        y = self.BN_down(F.relu(y, inplace=True))

        if self.unet is not None:
            if self.hp.pool:
                y_size = y.size()
                y, indices = self.pool(y, 2, stride=2, return_indices=True)
            y = self.unet(y)
            if self.hp.pool:
                y = self.unpool(y, indices, 2, stride=2, output_size=y_size)

        y = self.dropout(y)
        y = self.conv_up(y)
        y = self.BN_up(F.relu(y, inplace=True))
        y = torch.cat((x, y), 1)
        y = self.reduce(y)
        y = self.BN_out(F.relu(y, inplace=True))
        return y


class Unet1d(Unet):
    def __init__(self, hp: HyperparametersUnet):
        super().__init__(hp, nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d, F.max_pool1d, F.max_unpool1d)


class Unet2d(Unet):
    def __init__(self, hp: HyperparametersUnet):
        super().__init__(hp, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, F.max_pool2d, F.max_unpool2d)


class HyperparametersCatStack(Hyperparameters):
    def __init__(self, N_layers, kernel, padding, stride, **kwargs):
        super().__init__(**kwargs)
        self.N_layers = N_layers
        self.kernel = kernel
        self.padding = padding
        self.stride = stride


class ConvBlock(nn.Module):
    def __init__(self, hp: Hyperparameters, conv:ClassVar, bn: ClassVar):
        self.hp = hp
        super().__init__()
        self.dropout = nn.Dropout(self.hp.dropout_rate)
        self.conv = conv(self.hp.hidden_channels, self.hp.hidden_channels, self.hp.kernel, self.hp.stride, self.hp.padding)
        self.BN = bn(self.hp.hidden_channels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.BN(F.relu(x, inplace=True))
        return x


class CatStack(nn.Module):
    def __init__(self, hp: HyperparametersCatStack, conv: ClassVar, bn: ClassVar, conv_block: ClassVar):
        super().__init__()
        self.hp = hp
        self.conv_stack = nn.ModuleList()
        for i in range(self.hp.N_layers):
            self.conv_stack.append(conv_block(hp))
        self.reduce = conv((1 + self.hp.N_layers) * self.hp.hidden_channels, self.hp.hidden_channels, 1, 1)
        self.BN = bn(self.hp.hidden_channels)

    def forward(self, x):
        x_list = [x]
        for i in range(self.hp.N_layers):
            x = self.conv_stack[i](x)
            x_list.append(x)
        x = torch.cat(x_list, 1)
        x = self.reduce(x)
        y = self.BN(F.relu(x, inplace=True))
        return y


class ConvBlock1d(ConvBlock):
    def __init__(self, hp: HyperparametersCatStack):
        super().__init__(hp, nn.Conv1d,  nn.BatchNorm1d)


class CatStack1d(CatStack):

    def __init__(self, hp: HyperparametersCatStack):
        super().__init__(hp, nn.Conv1d,  nn.BatchNorm1d, ConvBlock1d)


class ConvBlock2d(ConvBlock):
    def __init__(self, hp: HyperparametersCatStack):
        super().__init__(hp, nn.Conv2d,  nn.BatchNorm2d)


class CatStack2d(CatStack):

    def __init__(self, hp: HyperparametersCatStack):
        super().__init__(hp, nn.Conv2d, nn.BatchNorm2d, ConvBlock2d)


def self_test():
    hpcs = HyperparametersCatStack(N_layers=2, kernel=7, padding=3, stride=1, in_channels=1, out_channels=1, hidden_channels=2, dropout_rate=0.1)
    cs2d = CatStack2d(hpcs)
    cb2d = ConvBlock2d(hpcs)
    cs1d = CatStack1d(hpcs)
    cs2d = ConvBlock1d(hpcs)
    hpun = HyperparametersUnet(nf_table=[2,2,2], kernel_table=[3,3], stride_table=[1,1,1], pool=True, in_channels=1, hidden_channels=2, out_channels=1, dropout_rate=0.1)
    un2d = Unet2d(hpun)
    c1dcs = Container1d(hpcs, CatStack1d)
    c2dcs = Container2d(hpcs, CatStack2d)
    c1dun = Container1d(hpun, Unet1d)
    c2dun = Container2d(hpun, Unet2d)

    print("It seems to work: classes could be instantiated")

def main():
    self_test()

if __name__ == '__main__':
    main()