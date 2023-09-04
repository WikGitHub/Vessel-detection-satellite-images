## Below are initial notes on the CNN architecture to be used for vessel segmentation.

## Goals

Develop and train a U-Net based CNN for vessel segmentation.

U-Net architecture was chosen due to its effectiveness in semantic segmentation tasks due to its symmetric encoder-decoder structure.

## Model complexity 

Initially, a basic U-Net implementation will be used, without any complex modifications.

## Input layer

The model will need to be able to take images of 80x80 pixels as input and produce pixel-wise segmentation masks of the same size output. 

## Convolutional layers (feature extraction)

Consider 1D/2D/3D convolutions depending on the data type (in this case, images).

## Pooling layers (down-sampling)

Possible pooling layers to consider:
- max pooling
- average pooling

## Fully connected/dense layers (classification)

Can be adjusted in terms of number of neurons, dropout rates, activation functions, etc.

## Activation functions (non-linearity)

Possible activation functions to consider:
- ReLU
- Sigmoid
- Softmax
- tanh
- Leaky ReLU (for vanishing gradient problem)

## Loss function and optimiser 

Possible loss functions to consider:
- categorical cross-entropy
- dice coefficient loss


This initial model will be used as a baseline for further improvements.

