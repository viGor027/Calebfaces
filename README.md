# Overview

The objective of the project was:  
1) to build a model capable of recognizing bald people in the photos,
2) for me to develop competencies of building such model, for which reason I selected the most imbalanced feature.


![feature balance chart](https://i.ibb.co/mS715SW/10-most-imbalanced-attributes.png)

## Tech stack
- tensorflow, keras
- pandas
- numpy
- opencv
- sklearn
- albumentations

## Process

To speed up the process I used the backbone of ConvNeXtTiny, with its weights already adjusted to imagenet dataset.
Initially, I explored class weights by training the network
on a smaller dataset with stratifying by feature we want to predict, manually adjusting weights.
Subsequently, I selected optimal learning rate using custom callback. I experimented with various classifiers
and decided to continue training with the most promising one; I made some of the top layers of the backbone trainable and continued training
using another custom callback to schedule a learning rate of the model in the span of an epoch
(due to one epoch taking excessively long time I decided to train a backbone for only one epoch for each part of it that was made trainable; see ```continue_training.py```). 

# Results

1 - bald  
0 - not bald  

![confusion matrix](https://i.ibb.co/1QYN4T1/confusion-matrix-best-continued-2.png)

# Conclusion

Looking at the confusion matrix we can clearly see that model requires further tweaking,
however in home environment with insufficient computing power finding optimal hyperparameters
and conducting training is not possible in a reasonable time.

## Dataset link

[Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

## How to train the model yourself

First in the directory of cloned repo install all required libraries via:

```pip install -r requirements.txt```

Due to high number of required packages using virtual environment is recommended.

Then download the dataset, and unpack it into ```data``` folder so your folder structure looks in the following way:

![structure](https://i.ibb.co/XSDTB1z/struktura.png)  

Images from the dataset should be unpacked directly to ```img_align_celeba``` (```images here``` is there just to point exact location - it is not a folder)

Next step is to run ```organize_data.py``` - it will organize images for you, finally you can train the model running ```train.py```
