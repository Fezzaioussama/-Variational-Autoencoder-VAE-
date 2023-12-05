# Variational Autoencoder (VAE) for MNIST Dataset

This repository contains the implementation of a Variational Autoencoder (VAE) for the MNIST dataset. VAEs are a type of autoencoder that extends the traditional autoencoder by introducing probabilistic principles, allowing for the generation of new data samples.

## Overview

Variational Autoencoders combine elements of both autoencoders and variational inference. They not only learn a compressed representation of input data but also model the uncertainty in this representation. In this work, we explore the application of VAEs to the MNIST dataset, generating new handwritten digit samples.

## Model Architecture

The VAE architecture includes both an encoder and a decoder, each consisting of neural network layers. The encoder maps input images to a probability distribution in the latent space, and the decoder generates samples from this distribution to reconstruct the input.
