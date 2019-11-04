#!/usr/bin/env bash

python setup.py sdist bdist_wheel
python -m twine upload dist/*
rm -r /home/ilja/Documents/Promotion/Project_Helpers/autoencoders/dist
rm -r /home/ilja/Documents/Promotion/Project_Helpers/autoencoders/build
rm -r /home/ilja/Documents/Promotion/Project_Helpers/autoencoders/autoencoder.egg-info