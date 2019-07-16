#!/usr/bin/env python
# encoding: utf-8
"""
models.py

Part for building auto-encoders.

Copyright (c) 2019, I. Manakov

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np
import torch as pt
import torch.nn as nn
from torch.nn.functional import relu


def identity(x):
    return x


class GlobalAveragePooling2d(nn.Module):
    """class for performing global average pooling on 2d feature maps"""
    def forward(self, x):
        """
        calculates the average of each feature map in the tensor

        :param x: input tensor of shape [batch, channels, height, width]
        :return: tensor that containes the average for each channel, [batch, channel_average]
        """
        return pt.mean(pt.mean(x, -1), -1)[..., None, None]


class GeneralConvolution(nn.Module):
    """class for combining padding, convolution, normalization and activation"""

    def __init__(self, current_channels, out_channels, kernel_size, stride,
                 padding=nn.ReplicationPad2d, norm=nn.InstanceNorm2d, activation=relu, convolution=nn.Conv2d, affine=False,
                 **kwargs):
        """

        :param current_channels: channels of the input tensor
        :param out_channels: channels of the output tensor
        :param kernel_size: kernel size of the convolution, either int or iterable
        :param stride: stride of the convolution, iterable
        :param padding: function that will be used for padding the input tensor, None for no padding, default=ReplicationPad2d
        :param norm: function that will be used for normalizing the output tensor, None for no normalization, default=InstanceNorm2d
        :param activation: activation function that will be applied after convolution, None for identity, default=relu
        :param convolution: function that will be used for the convolution, default=Conv2d
        :param affine: boolean that indicated whether bias will be used in the normalization
        :param kwargs: keyword arguments that will be passed to the convolution function
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        current_channels, out_channels = int(current_channels), int(out_channels)
        self.convolution = convolution(current_channels, out_channels, kernel_size, stride, **kwargs)

        self.padding = padding
        self.norm = norm(out_channels, affine=affine)
        self.activation = activation

    def forward(self, x):
        """
        apply padding, convolution, normalization and activation in that order

        :param x: input tensor in channels_first form
        :return: output tensor
        """

        out = x

        if self.padding:
            if len(self.kernel_size) == 3:
                ignore = (0,)  # in 3d case depth dimension will not be padded
            else:
                ignore = None
            out = self.pad(out, ignore)

        out = self.convolution(out)

        if self.norm:
            out = self.norm(out)

        if self.activation:
            out = self.activation(out)

        return out

    def pad(self, x, ignore=None):
        """
        pad input tensor so that shape is unaltered after convolution

        :param x: input tensor, channels_first
        :param ignore: spatial dimensions that should be ignored during padding
        :return: padded tensor
        """

        def calc_size(d, k, s):
            """
            calculate size of the dimension after convolution

            :param d: length of the dimension
            :param k: kernel size
            :param s: stride
            :return: dimension length after convolution
            """
            return (d-k)//s + 1

        in_size = list(x.shape[2:])  # consider only spatial dimensions
        kernel_size = list(self.kernel_size)
        stride = list(self.stride)

        # exclude ignored spatial dimensions
        if ignore:
            for ax in ignore:
                in_size.pop(ax)
                kernel_size.pop(ax)
                stride.pop(ax)

        # calculate output size after convolution and from there the amount of padding necessary
        out_size = [calc_size(d, k, s) for d, k, s in zip(in_size, kernel_size, stride)]
        padding = [i - s*o for i, o, s in zip(in_size, out_size, stride)]
        padding = [[int(p/2), int(round(p/2))] for p in padding]

        # fill the ignored spatial dimension with 0 for padding
        if ignore:
            for ax in ignore:
                padding.insert(ax, [0,0])

        padding.insert(-1, padding.pop(0))  # pytorch padding dimensions are in reverse order compared to convolution
        padding = tuple([p for ps in padding for p in ps])  # padding needs to be a flat array

        # instantiate padding if it is not already
        if type(self.padding) == type:
            self.padding = self.padding(padding)

        # change padding range if it has changed
        elif self.padding.padding != padding:
            self.padding = type(self.padding)(padding)

        return self.padding(x)


class ConvResize(nn.Module):
    """
    class that combines upsampling by interpolation with a convolutional layer
    """

    def __init__(self, current_channels, out_channels, kernel_size, stride, upsampling, padding, norm,
                 activation, convolution, affine, **kwargs):
        """

        :param current_channels: number of channels in the input tensor
        :param out_channels: desired number of channels in the output tensor
        :param kernel_size: kernel size of the convolution, int or iterable
        :param stride: stride of the convolution, int or iterable
        :param upsampling: tuple of upsampling factor (int or iterable) and mode (as in pt.nn.Upsample)
        :param padding: function that will be used for padding, None for no padding
        :param norm: function that will be used for normalization, None for no normalization
        :param activation: activation function that is applied element-wise to the output of the convolution
        :param convolution: function that will be used for the convolution operation
        :param affine: bool indicating whether the normalization uses bias
        :param kwargs: additional keyword arguments for the convolution function
        """
        super().__init__()
        factor, mode = upsampling
        if pt.__version__ > '1.0.1':
            factor = factor[0]
        self.convolution = GeneralConvolution(current_channels, out_channels, kernel_size, stride,
                                              padding, norm, activation, convolution, affine=affine, **kwargs)
        self.upsampling = nn.Upsample(scale_factor=factor, mode=mode)

    def forward(self, x):
        out = x
        out = self.upsampling(out)
        out = self.convolution(out)

        return out


class ConvResize2d(ConvResize):
    """convenience class for resize convolution on 2d inputs"""
    def __init__(self, current_channels, out_channels, kernel_size, stride, upsampling=((2, 2), 'bilinear'),
                 padding=nn.ReplicationPad2d, norm=nn.InstanceNorm2d, activation=relu, convolution=nn.Conv2d,
                 affine=False, **kwargs):
        super().__init__(current_channels, out_channels, kernel_size, stride, upsampling, padding, norm,
                         activation, convolution, affine, **kwargs)


class ConvResize3d(ConvResize):
    """
    class that performs resize convolution in width and height dimensions
    and transpose convolution in depth dimension
    """
    def __init__(self, current_channels, out_channels, kernel_size, stride, upsampling=((1, 2, 2), 'trilinear'),
                 padding=nn.ReplicationPad3d, norm=nn.InstanceNorm3d, activation=relu, convolution=nn.Conv3d,
                 affine=False, **kwargs):

        super().__init__(current_channels, out_channels, kernel_size, stride, upsampling, padding, norm,
                 activation, convolution, affine, **kwargs)
        self.transpose_convolution = nn.ConvTranspose3d(current_channels, current_channels,
                                                        kernel_size=(kernel_size[0], 1, 1), stride=(1, 1, 1))

    def forward(self, x):

        out = x
        out = self.upsampling(out)
        out = self.transpose_convolution(out)
        out = self.convolution(out)

        return out


class ResBlock(nn.Module):
    """class that implements a residual block"""
    def __init__(self, channels, n_convolutions, kernel_size, padding, norm, activation, convolution, affine, stride,
                 **kwargs):
        """

        :param channels: number of channels in the input tensor, these will be maintained throughout the block
        :param n_convolutions: number of convolutions that will be performed in this block
        :param kernel_size: kernel size of the convolution, int or iterable
        :param padding: function that will be used for padding the input, None for no padding
        :param norm: function that will be used for normaliztaion, None for no normalization
        :param activation: activation function that is applied element-wise on the output of each convolution
        :param convolution: function that will be used for the convolution
        :param affine: bool that indicates whether the normalization uses bias
        :param stride: stirdes of the convolution, int or iterable
        :param kwargs:
        """
        super().__init__()
        self.layers = []
        for depth_index in range(n_convolutions):
            self.layers.append(GeneralConvolution(channels, channels, kernel_size, stride, padding,
                                                  norm, activation, convolution, affine, **kwargs))
            self.add_module('conv{}'.format(depth_index + 1), self.layers[-1])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out + x


class ResBlock2d(ResBlock):
    """
    convenience class with default arguments that construct residual blocks for 2d inputs
    """
    def __init__(self, channels, n_convolutions, kernel_size,
                 padding=nn.ReplicationPad2d, norm=nn.InstanceNorm2d, activation=relu, convolution=nn.Conv2d,
                 affine=False, stride=(1, 1), **kwargs):

        super().__init__(channels, n_convolutions, kernel_size, padding, norm, activation, convolution, affine, stride,
                         **kwargs)


class ResBlock3d(ResBlock):
    """
    convenience class with default arguments that construct residual blocks for 3d inputs
    """
    def __init__(self, n_convolutions, channels, kernel_size,
                 padding=nn.ReplicationPad3d, norm=nn.InstanceNorm3d, activation=relu, convolution=nn.Conv3d,
                 affine=False, stride=(1,1,1), **kwargs):

        super().__init__(channels, n_convolutions, kernel_size, padding, norm, activation, convolution, affine, stride,
                         **kwargs)


class Flatten(nn.Module):
    """
    callable class that reshapes any tensor into shape (batch_size, -1)
    """
    def forward(self, x):
        return x.view(-1, np.prod(x.shape[1:]))


