# Manuscript

## Title slide

*Niklas starts presenting*

So i'm Niklas, this is Thony and William. Our project is classification of Image-net with convolutional neural networks.

## Agenda

The presentation will be split into three parts, the first concerning our project proposal and intention in the beginning of the project.

Then, will describe our method and how we went about to implement our methodology.

Lastly, we will present our results and some retrospective on the project.

## Project proposal

During the lecture concerning CNNs image-net was mentioned and we as a group was inspired to make an own implementation to compare our results to the existing ones.

So, the overarching purpose of the project was to produce competitive and acceptable results on tiny image-net in terms of accuracy.

Along with this we set out to implement techniques which we've learned in the course and also techniques from previous work made by other in the image-net competition.

## Research of the field

We begun looking at the state of the art contributions to the image-net competitions. Those consisted mainly of implementation touched upon in the course such as GoogLeNet, RezNet and VGGNet.

We decided to work with the VGG net implementation because of its straight forward appearance.

We also found that Stanford's course CS231 includes a project on tiny imageNet and used these reports as an inspiration when deciding on the final structure of the network. We will explain this in further detail and how we implemented it.

## Stack and Data

We made our implementation in python, we used pandas and numpy for data handling and augmentation. We then used Keras which is built on top of Tensorflow to take care of everything regarding the network and training.

The arrangement between this course and Google Cloud Services allowed us to utilise a modern high performance GPU, Tesla K80 to run our computations on.

The tiny image dataset consists of 100.000 images divided in 200 classes which results in 500 images per class in the training set. The validation set consists of 10.000 images.

Each image is 64 by 64 pixel which is a reduction of the original size of 224 by 224 pixels. The test set has no available labels but predictions must be uploaded and then a results is received by an online service.

*Change presenter here to Thony*

## CNN Architectures

We decided to try a few variations of the network to see what technique resulted in improvements. The variations are presented in this table.

The leftmost structure represents our most vanilla implementation and as we move to the right we add more tricks by changing the solver, adding batch normalisation, dropout, another initialiser and eventually testing with more connected layers.

We initialised a run to test each of these networks for 20 epochs.

## Initial results

The first results was... (click in overfitting meme here)

## Loss plots

The overfitting is evident by these plots representing three of the architectures we just showed. Looking at these we realised model A and B had no regularisation at all which means overfitting were to be expected.

Model C and D had Dropout in the fully connected layer but only at a 30% ratio which apparently was not sufficient enough.

## Model D

When looking at the achieved accuracy on the validation set we found that model D performed the best with 29% accuracy when predicting 1 class. When allowed to "guess" 5 classes the accuracy was 55%.

However, the loss plot indicates serious overfitting and the accuracy is not satisfactory. So we decided to try prevent overfitting and achieve better results.

## Improving the results

To constrain the overfitting and achieve better results we tinkered with:
1. Batch normalisation to enable a more stable learning
2. Applying more Dropout
3. L2 Regularisation to prevent weights to fit specifically to the training data

*Change presenter here to William*

## Prevent overfitting 1/2

More specifically:
1. We preprocessed the pictures by subtracting the feature-wise mean of the dataset from each datapoint
2. Added a L2 regularization at the last layer of convolutional stack, which we initialized to .001
3. Did testing with 3 fully connected layers instead and increased dropout to 70%

This resulted in less overfitting for the 20 first epochs as visualised in this plot. However, the tail of the graph hints of beginning to overfit and, as we can observe on the graph, the learning in somewhat unstable

## Prevent overfitting 2/2

We kept tinkering with the parameters and made some attempt to stabilize the learning process of the network. We decided to rely on the L2 term and the reduction of the covariance shift on the batches that is introduced by the batch normalization for the regularization of the network, so we removed the dropout after some experimentation.

## Last two configurations

This resulted in two new sets of configuration, shown here as E and F. They are very identical in their architecture, but for F we  decided to try without the fully connected layers and increased the L2 regularization(to 5*10^-3) and the learning rate decay(to to 1*10^-3). We also set the initial learning rate a bit lower, using the number 0.001 from the original paper that introduced the Adam algorithm.

## Final loss and accuracy 1/2

The first model (E) achieved these results. From the plots we can tell training is slightly more stable but the loss plot reveals that overfitting still occurs. This was trained for 60 epochs.

## Final loss and accuracy 2/2

The last model (F) showed stable learning and much less signs of overfitting. This was run for 100 epochs. It appears to still progress slowly and potentially could achieve better results if allowed to run for even longer.

## Accuracy

To summary the results interestingly we found our best accuracy in model D which, as we mentioned, showed a lot of overfitting. However, we reason that the last model (F) has a more stable result due to its smoother learning process, which would generalise better and could achieve better accuracy if allowed to train for a longer period of time.

## Discussion and retro

We run into a few obstacles during the implementations where the first concerned some of the pictures was black and white and the dimensionality therefore differed from the other images. If we had began visualising the data these could be detected as outliers and dealt with.

We also regret not using RezNet, because it has fewer parameters and has a much faster training time. This would have given us more time to find optimal parameters of the network, since the slow speed of the VGG-net made course searching very difficult. Also, because of the residuals we reason we could also have tried deeper networks without overfitting.

And instead of tinkering manually with the parameters a parameter-search might have done the trick for us in a more structured manner.

So to conclude the things we learning from the project, tuning CNNs is hard and more of an art-form than a precise science.

Thank you!
