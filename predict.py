#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Chi Hoang
# DATE CREATED: 09/09/2018
# REVISED DATE: 09/10/2018 <=(Date Revised - if any)
    # Remove --arch flag and load model from the checkpoint info 
# PURPOSE: Use a trained network to predict the class for an input image
'''
Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

E.g. 
python predict.py flowers/test/15/image_06369.jpg densenet169_pretrained_checkpoint.pth --top_k 3
python predict.py flowers/test/15/image_06369.jpg densenet169_pretrained_checkpoint.pth --category_names cat_to_name.json --gpu
python predict.py flowers/test/15/image_06369.jpg densenet169_pretrained_checkpoint.pth --gpu

python predict.py flowers/test/15/image_06369.jpg DenseNet_1664_[1000]_102_1_0.001_checkpoint.pth --top_k 3 --gpu
python predict.py flowers/test/15/image_06369.jpg checkpoints/VGG_25088_[1024]_102_1_0.001_checkpoint.pth --top_k 5 --gpu

NOTE: If the loaded model was trained using GPU the first two commands won't work without --gpu because of torch tensor type mismatch
'''
# Imports python modules
import argparse
from PIL import Image

import torch
import torch.nn.functional as F

import torch
from torch import nn
from torch import optim

import utilities
import fc_model

def main():
    in_arg = get_input_args()
    
    image_path = in_arg.input
    checkpoint_path = in_arg.checkpoint
    category_names = in_arg.category_names
    topk = in_arg.top_k
    device = 'gpu' if in_arg.gpu else 'cpu'
    
    # Load checkpoint and rebuild our model
    saved_model, checkpoint = fc_model.load_checkpoint(checkpoint_path)
    print('Model loaded successfully!')
    
    # Get index to class map
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v : k for k, v in class_to_idx.items()}
    
    # Predict input image
    probs, classes = predict(image_path, saved_model, idx_to_class, topk, device)
    
    cat_to_name = fc_model.map_labels(category_names)
    names = [cat_to_name[c] for c in classes]
    
    print('Probabilities (%)', [float(round(p * 100.0, 2)) for p in probs])
    print('Classes:', names)
    
    print("Top most likely class:", names[0])

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
    5 command line arguements are created:       
       input - Path to the image file to predict
       checkpoint - Path to checkpoint file
       category_names - Path to classes map file
       top_k - number of highest probabilities
       gpu - whether to utilize gpu to train
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to the image file to predict')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to classes map file')
    parser.add_argument('--top_k', type=int, default=3, help='Number of highest probabilities')
    parser.add_argument('--gpu', action='store_true', help='Utilize gpu to train')

    return parser.parse_args()

def predict(image_path, model, idx_to_class, topk=5, device='cpu'):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print("Predicting the top {} classes with {} pre-trained model | device={}.".format(topk, model.__class__.__name__, device))
    
    # load an image file    
    processed_image = utilities.process_image(image_path).squeeze()
    
    # turn off drop-out
    model.eval()
    # change to cuda
    if device == 'gpu':
        model = model.cuda()
    
    with torch.no_grad():
        if device == 'gpu':
            output = model(torch.from_numpy(processed_image).float().cuda().unsqueeze_(0))
        else:
            output = model(torch.from_numpy(processed_image).float().unsqueeze_(0))
    
    # Calculate the class probabilities (softmax) for image
    ps = F.softmax(output, dim=1)
    
    top = torch.topk(ps, topk)
    
    probs = top[0][0].cpu().numpy()
    classes = [idx_to_class[i] for i in top[1][0].cpu().numpy()]
    
    return probs, classes

# Call to main function to run the program
if __name__ == "__main__":
    main() 