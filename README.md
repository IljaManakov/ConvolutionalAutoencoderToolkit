# Introduction
This repository contains the tools necessary to flexibly build an autoencoder in pytorch. In the future some more investigative tools may be added. The main goal of this toolkit is to enable quick and flexible experimentation with convolutional autoencoders of a variety of architectures.

The implementation is such that the architecture of the autoencoder can be altered by passing different arguments. Tunable aspects are:
- number of layers
- number of residual blocks at each layer of the autoencoder
- functions used for downsampling and upsampling convolutions and convolutions in the residual blocks
- number of channels at each layer of the autoencoder
- activation function performed after each convolution
- symmetry (or lack thereof) of the encoder-decoder architecture
- etc

Some usefull wrappers and custom classes, such as ResidualBlock or GeneralConvolution, can be found in model_parts.py.
The file models.py is where the actual autoencoder classes are. It contains one base class as well as two extension for 2d and 3d data.

# Installation
The latest stable version can be obtained using `pip install autoencoder`.

Otherwise, you can download and use the files directly in your projects.

# Usage
The ConvAE base class expects parameters that specify the overall architecture (see documentation) and one function for the downsampling layer, upsampling layer and residual block.
Conv2dAE and Conv3dAE on the other hand provide an interface to easily create the aforementioned functions from parameters and create the autoencoder from there.
