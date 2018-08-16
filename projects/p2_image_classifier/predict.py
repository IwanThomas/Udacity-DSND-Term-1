import numpy as np
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from PIL import Image
import utility_functions as h


def predict(image_path, model, topk, device):
    '''
    Predicts the class (or classes) of an image 
    using a trained deep learning model.
    '''
    image = h.process_image(image_path)
    # add another dimension as PyTorch expects to see the batch size
    image = torch.from_numpy(np.expand_dims(image, axis=0))
    
    # prepare model and data
    image = image.to(device)
    model.to(device)
    model.eval()
    
    # make prediction and output probs and classes
    prediction = model.forward(image)
    topk = prediction.topk(topk)
    log_probs, classes = topk[0], topk[1]
    
    # take exponent of log probs to give probs
    probs = torch.exp(log_probs)
    return (probs, classes)  

def main(image_path, checkpoint, device, topk=1, category_name=None):
    model = h.load_checkpoint(checkpoint)
    probs, classes = predict(image_path, model, topk, device)
    
    # get probs and classes out of tensors
    probs = probs.cpu().detach().numpy()[0]
    classes = classes.cpu().numpy()[0]
    
    # map class number to name if JSON provided
    if category_name:
        mapping = h.load_mappings(category_name)
        classes = list(map(lambda x: mapping.get(str(x)), classes))
    
    return probs[0], classes
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('image_path',
                        type=str,
                        help='Path to Image')
    parser.add_argument('checkpoint',
                        type=str,
                        help='File storing trained model architecture')
    parser.add_argument('--topk',
                        type=int,
                        help='Print out the top K classes along with associated probabilities')
    parser.add_argument('--category_name',
                        type=str,
                        help='Filename of JSON File that maps class values to category names')
    parser.add_argument('--gpu',
                        action='store_true',
                        help='Flag indicating GPU should be used for inference')
    
    args = parser.parse_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    topk = args.topk if args.topk else 1
    category_name = args.category_name
    device = "cuda:0" if args.gpu == 'yes' and torch.cuda.is_available() else "cpu"
    print (main(image_path, checkpoint, device, topk, category_name))
