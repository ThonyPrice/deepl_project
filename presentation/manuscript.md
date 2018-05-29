# Manuscript

## Title slide

So i'm X_name, this is Y_name and Z_name. Our project is classification of Image-net with convolutional neural networks.

## Agenda

The presentation will be split into three parts, the first concerning our project proposal and intention in the beginning of the project.

Then, Y_name will describe our method and how we went about to implement our methodology.

Lastly, Z_name will present our results and some retrospective on the project.

## Project proposal

During the lecture concerning CNNs image-net was mentioned and we as a group was inspired to make an own implementation to compare our results to the existing ones.

So, the overarching purpose of the project was to produce competitive and acceptable results on tiny image-net in terms of accuracy.

Along with this we set out to implement techniques which we've learned in the course and also techniques from previous work made by other in the image-net competition.

## Research of the field

We begun looking at the state of the art contributions to the image-net competitions. Those consisted mainly of implementation touched upon in the course such as GoogLeNet, RezNet and VGGNet.

We decided to work with the VGG net implementation because of its straight forward appearance and higher accuracy than GoogLeNet and RezNet-18.

We also found that Stanford's course CS231 includes a project on tiny imageNet and used these reports as an inspiration when deciding on the final structure of the network. Name_Y will explain this in further detail and how we implemented it.

## Stack and Data

We made our implementation in python, we used pandas and numpy for data handling and augmentation. We then used Keras which is build on top of Tensorflow to take care of everything regarding the network and training.
GPU set up for computations.

The tiny image dataset consists of 100.000 images, 200 and 200 classes which results in 500 images per class in the training set. The validation set consists of 10.000 images.

## CNN Architectures

We decided to try a few variations of the network to see what technique resulted in improvements. The variations are presented in this table.

The leftmost structure represents our most vanilla implementation and as we move to the right we add more tricks by changing the solver, adding batch normalisation, dropout and eventually testing with more connected layers.

<!-- Below - Introduction to markdown -->

# Header

## Subsection

I suggest we write out manuscript here, in markdown.

Some *italic* and **bold** text.

A list:
1. Which
2. is
3. ordered

You can use markdown emojis too :smirk:

Find cheat sheet on this [link](https://gist.github.com/rxaviers/7360908)

Use Atom package 'Markdown preview plus' to render file
