import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoencoder",
    version="0.0.3",
    author="Ilja Manakov",
    author_email="ilja.manakov@gmx.de",
    description="A toolkit for flexibly building convolutional autoencoders in pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IljaManakov/ConvolutionalAutoencoderToolkit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
