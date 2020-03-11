#!/usr/bin/env bash

python setup.py sdist bdist_wheel
python -m twine upload dist/*
rm -r /home/ilja/projects/public/ConvolutionalAutoencoderToolkit/dist
rm -r /home/ilja/projects/public/ConvolutionalAutoencoderToolkit/build
rm -r /home/ilja/projects/public/ConvolutionalAutoencoderToolkit/autoencoder.egg-info