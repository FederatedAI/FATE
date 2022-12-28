# NN Modules Tutorial

In FATE-1.10，the whole NN framework is re-designed for highly customizable machine learning. 
With this framework, you can create your own models, datasets, trainers, and aggregators to meet your specific needs. 
This tutorial introduces you to our brand new framework. With links below, you can learn how to use our Homo & Hetero(Horizontal and Vertical) NN step by step. Please notice that this tutorial is based on the standalone version, if you are using the cluster version, you may have to deploy the codes/data 
on every party respectively.

## Quick Start

To get yourself be familiar with FATE-NN and pipeline, we recommend completing these two quick start tutorials. If you are using tabular data and do not require any customizations, these tutorials should be sufficient for your needs

- [Homo-NN Quick Start: A Binary Classification Task](Homo-NN-Quick-Start.ipynb)
- [Hetero-NN Quick Start: A Binary Classification Task](Hetero-NN-Quick-Start.ipynb)

## Homo-NN Customization
  
### 1.Dataset 

Customizing a dataset in PyTorch is similar to using the built-in PyTorch Dataset class. In this chapter, we will use a simple image classification task - the MNIST Handwritten Recognition dataset - as an example to demonstrate how to customize and use your own dataset in FATE.

- [Customize your Dataset](Homo-NN-Customize-your-Dataset.ipynb)
- [Using FATE Built-In Dataset](Introduce-Built-In-Dataset.ipynb)


### 2.Loss Function

In this section, we show you how to customize loss functions

- [Customize loss function](Homo-NN-Customize-Loss.ipynb)
  
### 3.Model

In FATE 1.10, you can create a PyTorch model by subclassing nn.Module. Using the example from the previous chapter, we will demonstrate how to develop, deploy, and use a more complex model in FATE.

- [Customize Model](Homo-NN-Customize-Model.ipynb)


### 4.Trainer

In order to show you how to develop your own Trainer, here we try to develop a simple Trainer to implement the FedProx algorithm (this is just an exsample, please do not use it for production). In addition, we will also show you how to use the interface that comes with TrainerBase to save the model, set the checkpoint, output the prediction results, and use the interface to display some data on FateBoard.

- [Customize trainer to control the training process](Homo-NN-Customize-Trainer.ipynb)

- [Using FATE-interfaces](Homo-NN-Trainer-Interfaces.ipynb)


### 5.Aggregator

- [Develop Aggregator: Introduce Basics](Homo-NN-aggregator.ipynb)


## Hetero-NN Customization

### Dataset

- [Customize your Dataset](Hetero-NN-Customize-Dataset.ipynb)

### Top Model, Bottom Model & Interactive Layer & Loss

- [Use CustModel to Set Top, Bottom Model](Hetero-NN-Customize-Model.ipynb)


## Advanced Examples

Here we offer some advanced examples of using FATE-NN framework.

### Resnet classification(Homo-NN)

- [Federated Rensnet training on CIFAR-10](Resnet-example.ipynb)

### Recommendation models(Homo-NN)

- [Federated classic CTR model training on Criteo](CTR-example.ipynb)

### Federated Text Classification Using Bert(Homo-NN)

- [Using Frozen Parameters Bert for Sentiment Classification](Bert-example.ipynb)

### Training on vertical-split heterogeneous data(Hetero-NN)

- [A federated task with guest using image data and host using text data](Hetero-NN-Mix-Task.ipynb)
