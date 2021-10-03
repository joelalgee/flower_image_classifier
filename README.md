# Image Classification with Pytorch - February 2021

Classifying images through a command line application, using neural networks in Pytorch

## Summary

In this project, I created a command line application that trains a neural network to classify flower images by species, and can then be used to predict which species a new flower image depicts. Although the images were flowers for this project, any set of images sorted into classes, with an associated json file to convert folder numbers to classes, could be used.

## Package versions

* python 3.8.5
* torch 1.7.0
* PIL 7.2.0
* numpy 1.19.1

## Instructions

### Setup

Save all files in the same directory. Create a subfolder for images, containing a directory tree as follows:
```
images
  train
    1
      class 1 image 1.jpg
      class 1 image 2.jpg
      class 1 image 3.jpg
      ...etc
    2
      class 2 image 1.jpg
      ...etc
    ...etc
  valid
    1
      class 1 image 4.jpg
      ...etc
    ...etc
  test
    1
      class 1 image 5.jpg
      ...etc
    ...etc
```
### Train

Train a new network on a data set with train.py

**Basic usage:** python train.py data_directory

Prints out training loss, validation loss, and validation accuracy as the network trains

**Options:**

* Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
* Choose architecture ("vgg13" or "squeezenet"): python train.py data_dir --arch "vgg13"
* Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
* Use GPU for training: python train.py data_dir --gpu

### Predict

Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

**Basic usage:** python predict.py /path/to/image checkpoint

**Options:**

* Return top KK most likely classes: python predict.py input checkpoint --top_k 3
* Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
* Use GPU for inference: python predict.py input checkpoint --gpu

## Files

### train.py

Code for training the model.

### predict.py

Code for making predictions.

### model_helper.py

Helper functions relating to the model.

### utility.py

Utility functions for loading data and preprocessing images.

### cat_to_name.json

A key for converting folder numbers to species names.

## Data

The dataset provided by Visual Geometry Group at University of Oxford can be downloaded from [this page](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

## Credits

This project was provided by [Udacity](https://www.udacity.com) as part of their [Intro to Machine Learning with Pytorch nanodegree](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229). Data were provided by [Visual Geometry Group at University of Oxford](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).
