'''
Functions and classes relating to the model
'''
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import json
from time import time, sleep, localtime, strftime

import utilities


def map_labels(labels_path):
    with open(labels_path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def pretrain(network):
    # Load a pre-trained network
    if network == "VGG":
        pretrained_model = models.vgg16(pretrained=True)
    elif network == "DenseNet":
        pretrained_model = models.densenet169(pretrained=True)
    else:
        raise Exception("Architecture not supported!")

    return pretrained_model


def build_model(arch, input_size, hidden_layers, output_size, drop_p=0.5):
    '''
    Define network architecture, build our model using the loaded pre-trained model
    '''

    # Load a pre-trained model
    model = pretrain(arch)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_layers[0])),
                              ('relu1', nn.ReLU()),
                              ('drop_out1', nn.Dropout(p=drop_p)),
                              ('fc2', nn.Linear(
                                  hidden_layers[0], output_size)),
                              ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


def train_model(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    '''
    Train a model with a pre-trained network
    '''

    steps = 0

    # turn on drop-out
    model.train()

    # train model with cuda if available
    device = torch.device("cuda:0" if device == 'gpu' else "cpu")
    model.to(device)

    print("Training with {} pre-trained model || epochs={} || device={}.".format(
        model.__class__.__name__, epochs, device))

    # Set up timer
    start_time = time()

    for e in range(epochs):
        running_loss = 0

        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Prints out training loss, validation loss, and validation accuracy as the network trains
            if steps % print_every == 0:
                test_loss, accuracy = validation(model, validloader, criterion)

                print("Device={} - Epoch: {}/{}... ".format(device, e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(
                          running_loss / print_every),
                      "Validation Loss: {:.3f}.. ".format(
                          test_loss / len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(validloader)))

                running_loss = 0

    end_time = time()
    tot_time = end_time - start_time
    tot_time = strftime('%H:%M:%S', localtime(tot_time))
    print("\n** Total Elapsed Training Runtime: ", tot_time)

# Validation pass


def validation(model, validloader, criterion):
    test_loss = 0
    accuracy = 0

    # test model with cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # turn off drop-out
    model.eval()

    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

    # Turn drop-out back on
    model.train()

    return test_loss, accuracy


def check_accuracy_on_test(model, testloader):
    '''
    Do validation on the test set
    '''
    correct = 0
    total = 0
    image_count = 0

    # test model with cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # turn off drop-out
    model.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            image_count += 1

    print('Accuracy of the network on {} test images: {:.2f}%'.format(
        image_count, 100 * correct / total))

# Load a checkpoint and rebuilds the model


def load_checkpoint(filepath):
    # Load the checkpoint
    # checkpoint = torch.load(filepath)
    checkpoint = torch.load(
        filepath, map_location=lambda storage, loc: storage)

    # Load the pre-trained model and rebuild our model
    model = pretrain(checkpoint['architecture'])
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    print('Checkpoint loaded successfully!')

    return model, checkpoint
