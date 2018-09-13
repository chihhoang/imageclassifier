# Image Classifier - Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, we'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories.

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

This is an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it.

Specifically, I have trained my models and saved them as checkpoints that can be used to predict species of a flower from an image. In the future, if we want to train on a different dataset, we will need to retrain our model using train.py. Basic usage to predict a flower image species can be found below.

# Main files
- predict.py - used to predict a flower species from an image
- train.py - used to train our own model using a pretrained classifier (only support VGG and DenseNet architecture for now)
- ImageClassifierProject.ipynb - the process to develop this project

# Dependencies (via conda recommended)
- Core Python 3 packages
- pytorch
- torchvision
- PIL (pillow)

E.g. tested on MacOS
```
$ conda install python=3 numpy pandas pillow pytorch torchvision -c pytorch
```

*I had a few package conflicts when installing the dependencies. The best way to solve these problems is to create a new environment with conda and install the packages there.*

### Below is the result of my model training on the flower testset with various architectures and hyperparameters
| Architecture | Epochs | Hidden Layers | Learning Rate | Accuracy | Model Saved                                     |
| ------------ | ------ | ------------- | ------------- | -------- | ----------------------------------------------- |
| VGG16        | 3      | [2048, 1024]  | 0.001         | 76.07%   |                                                 |
| VGG16        | 1      | [4096, 2048]  | 0.001         | 64.10%   |                                                 |
| VGG16        | 1      | [1, 1024]     | 0.001         | 0.49%    |                                                 |
| VGG16        | 1      | [1024, 2048]  | 0.001         | 1.34%    |                                                 |
| VGG16        | 2      | [1024, 2048]  | 0.001         | 1.46%    |                                                 |
| VGG16        | 4      | [1024]        | 0.001         | 82.78%   |                                                 |
| VGG16        | 5      | [1024]        | 0.001         | 85.47%   |                                                 |
| DenseNet169  | 5      | [1024]        | 0.001         | 93.52%   |                                                 |
| DenseNet169  | 5      | [1024]        | 0.001         | 92.06%   | densenet169_pretrained_checkpoint.pth           |
| DenseNet169  | 4      | [1024]        | 0.001         | 90.60%   | DenseNet_1664_[1024]_102_5_0.001_checkpoint.pth |


# Basic Usages
```
$ python predict.py [path to your image] [path to checkpoint] [optional flags]
```

--top_k - number of results with top highest probabilities  
--gpu - whether to utilize gpu to train

E.g.
```  
$ python predict.py img001.jpg DenseNet_1664_[1024]_102_5_0.001_checkpoint.pth --top_k 3
```

More examples can be found in predict.py

Outputs from running the above command:
```
Using CPU for calculations
...
Checkpoint loaded successfully!
Model loaded successfully!
Predicting the top 3 classes with DenseNet pre-trained model | device=cpu.
Probabilities (%) [96.33, 1.54, 0.36]
Classes: ['rose', 'sword lily', 'bromelia']
Top most likely class: rose

```
