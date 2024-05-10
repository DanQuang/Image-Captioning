# Image Captioning
This is the model for the Image Captioning task, using Pytorch

## About
Using pretrained model [Inception v3](https://pytorch.org/hub/pytorch_vision_inception_v3/) to encode the images and LSTM as decoder to generate captions from the image features.
This Example use `flickr8k` dataset.

## Usage
To train the Image Captioning model, use the command line:

``bash
python main.py --config config.yaml
``
