#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Chi Hoang
# DATE CREATED: 09/09/2018
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Train a new network on a dataset and save the model as a checkpoint
'''
Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

E.g.
Quick tests
python train.py flowers --gpu
python train.py flowers --arch "VGG" --save_dir checkpoints --learning_rate 0.002 --hidden_units 1024 --epochs 1 --gpu

Best accuracy so far
python train.py flowers --arch "DenseNet" --save_dir checkpoints --learning_rate 0.001 --hidden_units 1024 --epochs 5 --gpu
'''
# Imports python modules
import argparse
import os
from PIL import Image

import torch
from torch import nn
from torch import optim

import utilities
import fc_model

def main():
    
    in_arg = get_input_args()
    print('Received inputs from the command', in_arg)
    
    data_dir = in_arg.data_dir
    arch = in_arg.arch # only support "VGG" or "DenseNet"
    save_dir = in_arg.save_dir
    device = 'gpu' if in_arg.gpu else 'cpu'
    
    # Check inputs
    if arch not in ["VGG", "DenseNet"]:
        raise Exception(arch + " architecture not supported! Please use either 'VGG' or 'DenseNet' for --arch flag.")
    
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Load data
    train_data, trainloader, validloader, testloader = utilities.load_data(data_dir)

    # Load pretrain network
    pretrained_model = fc_model.pretrain(arch)
    
    # Get pretrained model name
    model_name = pretrained_model.__class__.__name__
    
    # Get hyperparameters from command line arguments
    epochs = in_arg.epochs
    if (model_name == "DenseNet"):
        input_size = pretrained_model.classifier.in_features
    elif (model_name == "VGG"):
        input_size = pretrained_model.classifier[0].in_features
    else:
        raise Exception("Architecture not supported!")
        
    hidden_layers = [in_arg.hidden_units]
    output_size = 102
    learning_rate = in_arg.learning_rate
    drop_p = 0.5
    
    # Build our model
    model = fc_model.build_model(arch, input_size, hidden_layers, output_size, drop_p)
    
    # Define our lost function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print('Training with hyperparams: epochs={}||input_size={}||hidden_layers={}||output_size={}||learning_rate={}||drop_p={}'.format(epochs, input_size,                        hidden_layers, output_size, learning_rate, drop_p))
    
    # Train our model
    print_every = 40
    fc_model.train_model(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device)
    fc_model.check_accuracy_on_test(model, testloader)
    
    # Save the checkpoint
    checkpoint = {'architecture': model.__class__.__name__,
                  'input_size': model.classifier[0].in_features,
                  'hidden_layers': hidden_layers,
                  'output_size': 102,
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'optimizer_state': optimizer.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx}
    
    # Create a folder to save checkpoint if not already existed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # file name format: {architecture}_{input_size}_{hidden_layers}_{output_size}_{epochs}_{learning_rate}_checkpoint.pth
    torch.save(checkpoint, save_dir + '/{}_{}_{}_{}_{}_{}_checkpoint.pth'.format(checkpoint['architecture'],
          checkpoint['input_size'], checkpoint['hidden_layers'], checkpoint['output_size'],
          checkpoint['epochs'], checkpoint['learning_rate']))
#     torch.save(checkpoint, 'densenet169_pretrained_checkpoint.pth')
    print('Model saved successfully!')
    
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
    7 command line arguements are created:
       data_dir - Path to the image files
       arch - pretrained CNN model architecture to use for image classification (default-
              pick any of the following vgg, densenet)
       save_dir - Set directory to save checkpoints
       learning_rate - learning rate for optimizer
       hidden_units - number of hidden units
       epochs - number of epochs
       gpu - whether to utilize gpu to train
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Path to data files')
    parser.add_argument('--arch', type=str, default='DenseNet', help='CNN model architecture to use for image classification')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--hidden_units', type=int, default=1000, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Utilize gpu to train')

    return parser.parse_args()

# Call to main function to run the program
if __name__ == "__main__":
    main()    
    
