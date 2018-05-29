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

We decided to work with the VGG net implementation because of its straight forward appearance and higher accuracy than GoogLeNet and RezNet-18. If we restrict ourselves to a reasonable amount of layers.

We also found that Stanford's course CS231 includes a project on tiny imageNet and used these reports as an inspiration when deciding on the final structure of the network. Name_Y will explain this in further detail and how we implemented it.

## Stack and Data

We made our implementation in python, we used pandas and numpy for data handling and augmentation. We then used Keras which is build on top of Tensorflow to take care of everything regarding the network and training.
The arrangement between this course and Google Cloud Services allowed us to utilise a modern high performance GPU, Tesla K80 to run our computations on.

The tiny image dataset consists of 100.000 images divided in 200 classes which results in 500 images per class in the training set. The validation set consists of 10.000 images.
Each image is 64 by 64 pixel which is a reduction of the original size of 224 by 224 pixels. The test set has no available labels but predictions must be uploaded and then a results is received.

## CNN Architectures

We decided to try a few variations of the network to see what technique resulted in improvements. The variations are presented in this table.

The leftmost structure represents our most vanilla implementation and as we move to the right we add more tricks by changing the solver, adding batch normalisation, dropout, another initialiser and eventually testing with more connected layers.

We initialised a run to test each of these networks for 20 epochs, name_Z will tell you about the results.

## Initial results

The first results was... (click in overfitting meme here)

## Loss plots

The overfitting is evident by these lots representing three of the architectures we just showed. Looking at these we realised model A and B had no regularisation at all which means these results were to be expected.

Model C and D had Dropout in the fully connected layer but only at a 30% ratio which apparently was not sufficient enough.

## Model D

When looking at the achieved accuracy on the validation set we found that model D performed the best with 29% accuracy when predicting 1 class. When allowed to "guess" 5 classes the accuracy was 55%.

However, the loss plot indicates serious overfitting and the accuracy is not satisfactory to us so we decided to try prevent overfitting and achieve better results.

## War on overfitting

To constrain the overfitting we tinkered with:
1. Batch normalisation to enable a more stable learning
2. Applying more Dropout
3. L2 Regularisation to prevent weights to fit specifically to the training data

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
