# 3D-echo-generation

Repository for code from the paper "A Data Augmentation Pipeline to Generate Synthetic Labeled Datasets of 3D Echocardiography Images Using a GAN", available [here](https://ieeexplore.ieee.org/abstract/document/9893790).


## Overview

This repository contains the scripts to train a 3D GAN, in order to generate 3D echocardiography images. 

The generative model relies on a 3D version of the original [pix2pix](https://github.com/phillipi/pix2pix) model, using a 3D U-Net.

At inference time, the input is an anatomical mask of different cardiac structures and the output is the corresponding synthetic 3D echocardiography image.

![3D echocardiography generation](figures/diagram_IEEE_access_July22.png)
