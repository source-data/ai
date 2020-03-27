import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Callable, ClassVar
from collections import namedtuple
from copy import deepcopy
from .nvidia import PartialConv2d as PC2D


class PartiaLConv2d (nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ones = torch.ones_like(self.weight)
        self.n = self.ones.size(1) * self.ones.size(2) * self.ones.size(3)
        self.mask_conv = F.conv2d

    def forward(self, input, mask=None):
        assert input.dim() == 4
        if mask is not None:
            mask_for_input = mask.repeat(1, input.size(1), 1, 1) # same number of features as input
            output = super().forward(input * mask_for_input)
            with torch.no_grad():
                W = self.ones.to(input) # to move to same cuda device as input when necessary
                mask_for_output = self.mask_conv(mask_for_input, W, bias=None, padding=self.padding, stride=self.stride)
                ratio = self.n / (mask_for_output + 1e-8) # 1 where mask is 1, 10e8 where mask is zero
                mask_for_output = mask_for_output.clamp(0, 1) 
                ratio = ratio * mask_for_output # remove the 10e8 and keep the ones
                bias_view = self.bias.view(1, self.out_channels, 1, 1)
                output = ((output - bias_view) * ratio) + bias_view
                output = output * mask_for_output # is this necessary? ratio is zero anyway where mask is zero...
                new_mask = mask_for_output.sum(1).clamp(0, 1)
                new_mask = new_mask.unsqueeze(1)
        else:
            output = super().forward(input)
            new_mask = None
        assert output.dim() == 4
        return output, new_mask


class PartiaLTransposeConv2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ones = torch.ones_like(self.weight)
        self.mask_conv = F.conv_transpose2d

    def forward(self, input, mask=None):
        if mask is not None:
            mask_for_input = mask.repeat(1, input.size(1), 1, 1) # same number of features as input
            output = super().forward(input * mask_for_input)
            with torch.no_grad():
                W = self.ones.to(input) # to move to same cuda device as input when necessary
                mask_for_output = self.mask_conv(mask_for_input, W, padding=self.padding, stride=self.stride)
                # do we really need to scale by ratio of true pixels?
                # mask = mask.nelement() / mask.sum()
                output = output * mask_for_output
                new_mask = mask_for_output.sum(1).clamp(0, 1)
                new_mask = new_mask.unsqueeze(1)
        else:
            output = super().forward(input)
            new_mask = None
        return output, new_mask



class Hyperparameters:
    """
    The base class to hold model hyperparameters.

    Params and Attributes:
        in_channels (int): the number of input channels (or features).
        hidden_channels (int): the number of channels of the hidden layers.
        out_channels (int): the number of output channesl (or features)
        dropout_rate (float): dropout rate during training
    """

    def __init__(self, in_channels: int=None, hidden_channels: int=None, out_channels: int=None, dropout_rate: float=None):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

    def __str__(self):
        return "; ".join([f"{a}={v}" for a, v in self.__dict__.items()])


class Container(nn.Module):
    """
    Base class for 1D or 2D Container models. This class is not meant to be instantiated, only its subclasses.
    It includes an adapter layer that adapts the input channels the desired number of hidden channels.
    It includes a compression layer that adapts the hidden channels to the desired number of output channels. 
    Adapter and compression layers have a batch normalization of ReLU;

    Params:
        hp (Hyperparameters): the hyperparameters of the model.
        model (ClassVar): the class of the internal model of the Container (Unet or CatStack, 1d or 2d versions).
        conv (ClassVar): the class of the convolution (nn.Conv1d or nn.Conv2d) used for the adapter and compression layers.
        bn (ClassVar): the class of the BatchNorm layer (nn.BatchNorm1d or nn.BatchNorm2d)
    """

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
        x = self.BN_adapt(F.elu(x, inplace=True))
        z = self.model(x)
        z = self.compress(z)
        z = self.BN_out(F.elu(z, inplace=True)) # need to try without to see if it messes up average gray level
        return z


class Container1d(Container):
    """
    1D Container model. 

    Params:
        hp (Hyperparameters): the hyperparameters of the model.
        model (ClassVar): the class of the internal model of the Container.
    """

    def __init__(self, hp: Hyperparameters, model: ClassVar):
        super().__init__(hp, model, nn.Conv1d,  nn.BatchNorm1d)


class Container2d(Container):
    """
    2D Container model. 

    Params:
        hp (Hyperparameters): the hyperparameters of the model.
        model (ClassVar): the class of the internal model of the Container.
    """

    def __init__(self, hp: Hyperparameters, model: ClassVar):
        super().__init__(hp, model, nn.Conv2d,  nn.BatchNorm2d)


class Container2dPC(nn.Module):
    """
   

    Params:
        hp (Hyperparameters): the hyperparameters of the model.
    """

    def __init__(self, hp: Hyperparameters):
        super().__init__()
        self.hp = hp
        self.out_channels = self.hp.out_channels
        self.adaptor = nn.Conv2d(self.hp.in_channels, self.hp.hidden_channels, 1, 1)
        self.BN_adapt = nn.BatchNorm2d(self.hp.hidden_channels)
        self.model = CatStack2dPC(self.hp) # Unet2dPC(self.hp) # 
        self.compress = nn.Conv2d(self.hp.hidden_channels, self.hp.out_channels, 1, 1)
        self.BN_out = nn.BatchNorm2d(self.hp.out_channels)

    def forward(self, x, mask=None):
        x = self.adaptor(x)
        x = self.BN_adapt(F.elu(x, inplace=True))
        z = self.model(x, mask) # z, = self.model(x, mask) if unet
        z = self.compress(z)
        z = self.BN_out(F.elu(z, inplace=True)) # need to try without to see if it messes up average gray level
        return z


class HyperparametersUnet(Hyperparameters):
    """
    Hyperparameters for U-net models. Extends the base class Hyperparameters.
    The number of layers (depth) of the U-net is simply specified by giving appropriate list of channels, kernels and stride.
    
    Usage: a 3 layers U-net is specified with the following hyperparameters
        HyperparametersUnet(
            nf_table=[2,4,8, 16], # 1st layer: 2 -> 4 channels, 2nd layer: 4 -> 8 channels; 3rd layer: 8 -> 16 channels.
            kernel_table=[3,3,3], # the three layers use same kernel 3
            stride_table=[1,1,1], # the three layers use the same stride 1
            pool=True, # pooling switched on
            in_channels=2, # params from base class
            hidden_channels=2, # params from base class
            out_channels=3, # params from base class
            dropout_rate=0.1 # params from base class
        )

    Params and Attributes:
        nf_table (List[int]): the number of channels (features) of each layers in the form [in_channels, out/in_channels, in/out_channels, in/out_channels, ...]
        kernel_table (List[int]): a list of the kernels for each layer.
        stride_table (List[int]): a list of the strides for each layer.
        pool (bool): indicate whether to include a pool/unpool step between layers.
    """

    def __init__(self, nf_table: List[int], kernel_table: List[int], stride_table: List[int], pool:bool, **kwargs):
        super().__init__(**kwargs)
        self.hidden_channels = nf_table[0]
        self.nf_table = nf_table
        self.kernel_table = kernel_table
        self.stride_table = stride_table
        self.pool = pool


class Unet(nn.Module):
    """
    Base class of 1D or 2D U-net models. This class is not meant to be instantiated.
    The U-net is built recursively. The kernel, padding and number of features of each layer is provided as lists in the HyperparamterUnet object.

    Params:
        hp (HyperparameterUnet): the model hyperparameters.
        model (ClassVar): the class of the internal model of the Container (Unet or CatStack, 1d or 2d versions).
        conv (ClassVar): the class of the convolution (nn.Conv1d or nn.Conv2d) used for the descending branch of the U-net.
        convT (ClassVar): the class of the transpose convolution (nn.ConvTranspose1d or nn.ConvTranspose2d) used for the ascending branch of the U-net.
        bn (ClassVar): the class of the BatchNorm layer (nn.BatchNorm1d or nn.BatchNorm2d).
        pool (ClassVar): the class of the pooling layer (nn.Pool1d or nn.Pool2d).
        unpool (ClassVar): the class of the unpooling layer (nn.Pool1d or nn.Pool2d).
    """

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
        y = self.BN_down(F.elu(y, inplace=True))

        if self.unet is not None:
            if self.hp.pool:
                y_size = y.size()
                y, indices = self.pool(y, 2, stride=2, return_indices=True)
            y = self.unet(y)
            if self.hp.pool:
                y = self.unpool(y, indices, 2, stride=2, output_size=list(y_size)) # list(y_size) is to fix a bug in torch 1.0.1; not need in 1.4.0

        y = self.dropout(y)
        y = self.conv_up(y)
        y = self.BN_up(F.elu(y, inplace=True))
        y = torch.cat((x, y), 1)
        y = self.reduce(y)
        y = self.BN_out(F.elu(y, inplace=True))
        return y


class Unet2dPC(nn.Module):
    """
    Base class of 1D or 2D U-net models. This class is not meant to be instantiated.
    The U-net is built recursively. The kernel, padding and number of features of each layer is provided as lists in the HyperparamterUnet object.

    Params:
        hp (HyperparameterUnet): the model hyperparameters.
    """

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
        self.conv_down = PartiaLConv2d(self.nf_input, self.nf_output, self.kernel, self.stride)
        self.BN_down = nn.BatchNorm2d(self.nf_output)
        self.pool = F.max_pool2d
        self.conv_up = nn.ConvTranspose2d(self.nf_output, self.nf_input, self.kernel, self.stride)
        self.unpool = F.max_unpool2d
        self.BN_up = nn.BatchNorm2d(self.nf_input)

        if len(self.hp.nf_table) > 1:
            self.unet = self.__class__(self.hp)
        else:
            self.unet = None

        self.reduce = nn.Conv2d(2*self.nf_input, self.nf_input, 1, 1)
        self.BN_out = nn.BatchNorm2d(self.nf_input)


    def forward(self, x, mask=None):

        y = self.dropout(x)
        y, new_mask = self.conv_down(y, mask)
        y = self.BN_down(F.elu(y, inplace=True))

        if self.unet is not None:
            if self.hp.pool:
                y_size = y.size()
                y, indices = self.pool(y, 2, stride=2, return_indices=True)
                if new_mask is not None:
                    new_mask_size = new_mask.size()
                    new_mask, mask_indices = self.pool(new_mask, 2, stride=2, return_indices=True)
            y, _ = self.unet(y, new_mask)
            if self.hp.pool:
                y = self.unpool(y, indices, 2, stride=2, output_size=list(y_size)) # list(y_size) is to fix a bug in torch 1.0.1; not need in 1.4.0
                if new_mask is not None:
                    new_mask = self.unpool(new_mask, mask_indices, 2, stride=2, output_size=list(new_mask_size))
        y = self.dropout(y)
        y = self.conv_up(y)
        y = self.BN_up(F.elu(y, inplace=True))
        y = torch.cat((x, y), 1)
        y = self.reduce(y)
        y = self.BN_out(F.elu(y, inplace=True))
        return y, mask

class Unet1d(Unet):
    """
    1D U-net. 

    Params:
        hp (HyperparametersUnet): the U-net hyperparameters.
    """

    def __init__(self, hp: HyperparametersUnet):
        super().__init__(hp, nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d, F.max_pool1d, F.max_unpool1d)


class Unet2d(Unet):
    """
    2D U-net. 

    Params:
        hp (HyperparametersUnet): the U-net hyperparameters.
    """

    def __init__(self, hp: HyperparametersUnet):
        super().__init__(hp, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, F.max_pool2d, F.max_unpool2d)


class HyperparametersCatStack(Hyperparameters):
    """
    Hyperparameters for CatStack models. Extends the base class Hyperparameters.
    
    Usage: a 3 layers U-net is specified with the following hyperparameters
        HyperparametersUnet(
            N_layers=2, 
            kernel=7, 
            padding=3, 
            stride=1
            in_channels=2, # params from base class
            hidden_channels=2, # params from base class
            out_channels=3, # params from base class
            dropout_rate=0.1 # params from base class
        )

    Params and Attributes:
        N_layers (int): the number of layers.
        kernel (int): kernel of the convoclution step.
        padding (int): padding added to each convolution.
        stride (int): stride of the convolution.
    """

    def __init__(self, N_layers, kernel, padding, stride, **kwargs):
        super().__init__(**kwargs)
        self.N_layers = N_layers
        self.kernel = kernel
        self.padding = padding
        self.stride = stride


class ConvBlock(nn.Module):
    """
    Base class of a convolution block including a batchnomr of ReLU.
    This class is not meant to be instantiated.
    
    Params:
        hp (Hyperparameters):
        conv (ClassVar): the class of the convolution (nn.Conv1d or nn.Conv2d) used for the adapter and compression layers.
        bn (ClassVar): the class of the BatchNorm layer (nn.BatchNorm1d or nn.BatchNorm2d)
    """

    def __init__(self, hp: Hyperparameters, conv:ClassVar, bn: ClassVar):
        self.hp = hp
        super().__init__()
        self.dropout = nn.Dropout(self.hp.dropout_rate)
        self.conv = conv(self.hp.hidden_channels, self.hp.hidden_channels, self.hp.kernel, self.hp.stride, self.hp.padding)
        self.BN = bn(self.hp.hidden_channels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.BN(F.elu(x, inplace=True))
        return x

class ConvBlock2dPC(nn.Module):
    """
    Base class of a partial convolution block including a batchnomr of ReLU.
    
    Params:
        hp (Hyperparameters):
    """

    def __init__(self, hp: Hyperparameters):
        self.hp = hp
        super().__init__()
        self.dropout = nn.Dropout(self.hp.dropout_rate)
        self.conv = PC2D(self.hp.hidden_channels, self.hp.hidden_channels, self.hp.kernel, self.hp.stride, self.hp.padding, multi_channel=True, return_mask=True)
        self.BN = nn.BatchNorm2d(self.hp.hidden_channels)

    def forward(self, x, mask=None):
        x = self.dropout(x)
        x, mask = self.conv(x, mask)
        x = self.BN(F.elu(x, inplace=True))
        assert x.dim() == 4
        return x, mask


class CatStack2dPC(nn.Module):
    """
    Base class for a CatStack model made of a concatenated stack of convolution blocks.

    Params:
        hp (HyperparametersCatStack)
    """

    def __init__(self, hp: HyperparametersCatStack):
        super().__init__()
        self.hp = hp
        self.conv_stack = nn.ModuleList()
        for i in range(self.hp.N_layers):
            self.conv_stack.append(ConvBlock2dPC(hp))
        self.reduce = nn.Conv2d((1 + self.hp.N_layers) * self.hp.hidden_channels, self.hp.hidden_channels, 1, 1)
        self.BN = nn.BatchNorm2d(self.hp.hidden_channels)

    def forward(self, x, mask=None):
        x_list = [x]
        for i in range(self.hp.N_layers):
            x, mask  = self.conv_stack[i](x, mask)
            x_list.append(x)
        x = torch.cat(x_list, 1)
        x = self.reduce(x)
        y = self.BN(F.elu(x, inplace=True))
        assert y.dim() == 4
        return y



class CatStack(nn.Module):
    """
    Base class for a CatStack model made of a concatenated stack of convolution blocks. This class is not meant to be instantiated.

    Params:
        hp (HyperparametersCatStack):
        conv (ClassVar): the class of the convolution (nn.Conv1d or nn.Conv2d) used for the adapter and compression layers.
        bn (ClassVar): the class of the BatchNorm layer (nn.BatchNorm1d or nn.BatchNorm2d)
        conv_block (ClassVar): the class of convolution block to build the stack. 
    """

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
        y = self.BN(F.elu(x, inplace=True))
        return y


class Autoencoder1d(nn.Module):
    """
    An 1D autoencoder based on a Container1d with CatStack1d. 
    The Container1d layer is followed by a reduction layer that addpats the number of output channels 
    to be equal to the number of input channels.

    Params:
        hp (HyperParametersCatStack): hyperparameters of the internal CatStack1d model.
    """

    def __init__(self, hp: HyperparametersCatStack):
        super(Autoencoder1d, self).__init__()
        self.in_channels = hp.in_channels
        self.hp = hp
        self.embed = Container1d(hp=self.hp, model=CatStack1d)
        self.reduce = nn.Conv1d(self.embed.out_channels, self.in_channels, 1, 1)
        
    def forward(self, x):
        y = self.embed(x)
        y = self.reduce(F.elu(y))
        return y


class ConvBlock1d(ConvBlock):
    """
    A 1D convolution block.

    Params:
        hp (HyperparametersUnet): the U-net hyperparameters.
    """

    def __init__(self, hp: HyperparametersCatStack):
        super().__init__(hp, nn.Conv1d,  nn.BatchNorm1d)


class CatStack1d(CatStack):
    """
    A 1D stracked convolution CatStack model.

    Params:
        hp (HyperparametersUnet): hyperparameters.
    """

    def __init__(self, hp: HyperparametersCatStack):
        super().__init__(hp, nn.Conv1d,  nn.BatchNorm1d, ConvBlock1d)


class ConvBlock2d(ConvBlock):
    """
    A 2D convolution block.

    Params:
        hp (HyperparametersUnet): hyperparameters.
    """

    def __init__(self, hp: HyperparametersCatStack):
        super().__init__(hp, nn.Conv2d,  nn.BatchNorm2d)


class CatStack2d(CatStack):
    """
    A 2D stracked convolution CatStack model.

    Params:
        hp (HyperparametersUnet): hyperparameters.
    """

    def __init__(self, hp: HyperparametersCatStack):
        super().__init__(hp, nn.Conv2d, nn.BatchNorm2d, ConvBlock2d)


def self_test():
    hpcs = HyperparametersCatStack(N_layers=2, kernel=7, padding=3, stride=1, in_channels=2, out_channels=3, hidden_channels=2, dropout_rate=0.1)
    cs2d = CatStack2d(hpcs)
    cb2d = ConvBlock2d(hpcs)
    cs1d = CatStack1d(hpcs)
    cb1d = ConvBlock1d(hpcs)
    hpun = HyperparametersUnet(nf_table=[2,2,2], kernel_table=[3,3], stride_table=[1,1,1], pool=True, in_channels=2, hidden_channels=2, out_channels=3, dropout_rate=0.1)
    un2d = Unet2d(hpun)
    c1dcs = Container1d(hpcs, CatStack1d)
    c2dcs = Container2d(hpcs, CatStack2d)
    c2dcs_PC = Container2dPC(hpcs)
    c1dun = Container1d(hpun, Unet1d)
    c2dun = Container2d(hpun, Unet2d)

    x1d = torch.ones(2, hpcs.in_channels, 100)
    x2d = torch.ones(2, hpcs.in_channels, 256, 256)
    mask = torch.ones(2, 2, 256, 256)
    cs1d(x1d)
    cs2d(x2d)
    cb1d(x1d)
    cb2d(x2d)
    c1dcs(x1d)
    c2dcs(x2d)
    c2dcs_PC(x2d, mask)
    c2dcs_PC(x2d)
    c1dun(x1d)
    c2dun(x2d)


    print("It seems to work: all classes could be instantiated and input forwarded.")

def main():
    self_test()

if __name__ == '__main__':
    main()