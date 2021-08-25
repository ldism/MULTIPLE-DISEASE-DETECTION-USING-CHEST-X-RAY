# MULTIPLE-DISEASE-DETECTION-USING-CHEST-X-RAY

The problem addressed:-
 To develop a deep learning model which can predict multiple diseases using chest X-rays
 
 Total 3 datasets are used to train the model:
 
1). NIH Chest X-rays  https://www.kaggle.com/nih-chest-xrays/data

2). VinBigData Original Image Dataset  https://www.kaggle.com/awsaf49/vinbigdata-original-image-dataset

3). Chest X-Ray Images (Pneumonia)  https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Data handling/processing:

Our model classifies 10 different diseases which are as follows:
1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Infiltration
5. Nodule/Mass
6. Pleural effusion
7. Pleural thickening
8. Pneumothorax
9. Pulmonary fibrosis
10. Pneumonia

To Prevent Data Leakage  In our data splitting, we have ensured that the split is done on the patient level so that there is no data "leakage" between the train and validation, datasets.

We also used the Keras generator to transform the values in each batch of images so that their mean is 0 and their standard deviation is 1.This will facilitate model training by standardizing the input distribution.
The generator also converts our single channel X-ray images (gray-scale) to a three-channel format by repeating the values in the image across all channels.
We will want this because the pre-trained model (on Imagenet) that we'll use requires three-channel inputs.

Model design

Abstract:
We have used two completely different models and trained them individually on each dataset.
After training individual models we have stacked them together by using ensembling techniques.

We develop an algorithm that can detect 10 different lung-related diseases classes from chest X-rays at a level comparable to practising radiologists. Our algorithm is a Combination of a 121-layer convolutional neural network based on Densenet-121 architecture and another 152-layer convolutional neural network based on Resnet-152 architecture, trained on about 200000 chest x rays from the source mentioned above.
Our final model is a stacked version of  Densenet-121 and Resnet-152 that inputs a chest X-ray image and outputs the probability of 10 different classes of lungs related diseases.

Problem Formulation:
The diseases detection task is a Multi-Class Classification. The problem, where the input is a frontal-view chest X-ray image X and the output is a vector of 10 elements each representing a binary label y ∈ {0, 1} indicating the absence or presence of a specific disease.
 For a single example in the training set, we optimize the weighted cross-entropy loss 

     L(X, y) = −(w+) . y log p(Y = 1|X) − (w−) ·(1 − y) log p(Y = 0|X), 

      Where
                  p(Y = i|X) is the probability that the network assigns to the label i, 
                  w+ = |N|/(|P|+|N|), and  
                  w− = |P|/(|P|+|N|) 
with |P| and |N| the number of positive cases and negative cases of the disease Y in the training set respectively.

Model Architecture and Training:



Model 1- DenseNet (121) based Architecture:

This is a 121-layer Dense Convolutional Network (DenseNet)  having pertained weight on the Imagenet dataset and afterwards trained on multiple chest X-ray datasets. DenseNets improve the flow of information and gradients through the network, making the optimization of very deep networks tractable. We replace the final fully-connected layer with one that has 10 outputs, after which we apply a sigmoid nonlinearity. The weights of the network are initialized with weights from a model pre-trained on ImageNet. The network is trained end-to-end using Adam with standard parameters (β1 = 0.9 and β2 = 0.999). We train the model using mini-batches of size 16. We use an initial learning rate of 0.001 and pick the model with the lowest validation loss. 


Model 2- ResNet (152) based Architecture:
Deep networks are hard to train because of the notorious vanishing gradient problem — as the gradient is back-propagated to earlier layers, repeated multiplication may make the gradient infinitely small. As a result, as the network goes deeper, its performance gets saturated or even starts degrading rapidly.
ResNet introduced a so-called “identity shortcut connection” that skips one or more layers, as shown in the following figure:


We replace the final fully-connected layer with one that has 10 outputs, after which we apply a sigmoid nonlinearity. The weights of the network are initialized with weights from a model pre-trained on ImageNet. The network is trained end-to-end using Adam with standard parameters (β1 = 0.9 and β2 = 0.999). We train the model using mini-batches of size 16. We use an initial learning rate of 0.001 and pick the model with the lowest validation loss. 


Model 3- Final Stacked model:
After training the two models separately on our datasets, we merged them and add another  Dense layer of 16 neurons and placed an output layer of 10 neurons after which we apply a sigmoid nonlinearity.
After that, we froze the input model layers and only train the last two added layers.

Detailed Results of each model can be found in the shared notebook.
trained weights are also made public.


Finally we have deployed the model as an app using streamlit!!!!!!!






