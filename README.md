# Overview

Idea of the project was to build a model capable of recognizing bald people
on the photos - for more challenging task the most imbalanced feature was selected,
and practice general process of making model.

![feature balance chart](https://i.ibb.co/mS715SW/10-most-imbalanced-attributes.png)

## Tech stack
- tensorflow, keras
- pandas
- numpy
- opencv
- sklearn
- albumentations

## Process

In this project was used the backbone of ConvNeXtTiny network.
At first, I was searching for optimal class weights, training the network
on a smaller dataset with stratifying by feature we want to predict, manually changing weights,
then the learning rate was adjusted. The whole hyperparameters adjusting took about training thirteen models.

# Results

1 - bald  
0 - not bald  

![confusion matrix](https://i.ibb.co/7JqSHHZ/confusion-matrix.png)

# Conclusion

Looking at the confusion matrix we can clearly see that model requires further tweaking,
however in home environment(insufficient computing power) optimal hyperparameters cannot be found in reasonable time span.

## Dataset link

[Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
