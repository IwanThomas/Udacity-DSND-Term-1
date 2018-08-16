import json
import torch
from PIL import Image
from torchvision import transforms, datasets, models
from train import Network

def load_mappings(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
def load_data(data_dir):
    "Returns generators for training, validation and test data"
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(100),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([
        transforms.Resize((224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    validate_transforms = transforms.Compose([
        transforms.Resize((224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, train_transforms)
    validate_data = datasets.ImageFolder(valid_dir, validate_transforms)
    test_data = datasets.ImageFolder(test_dir, test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validateloader = torch.utils.data.DataLoader(validate_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader, validateloader, testloader

def load_checkpoint(path):
    "Load saved network"
    checkpoint = torch.load(path)
    # instantiate and parameterize model
    model = Network(input_size=checkpoint['input_size'],
                    hidden_layers=checkpoint['hidden_layers'],
                    output_size=checkpoint['output_size'],
                    p_drop=checkpoint['p_drop'])

    model.load_state_dict(checkpoint['state_dict'])

    # append this to our pretrained model
    pretrained = models.vgg19(pretrained=True)
    pretrained.classifier = model

    return pretrained

def save_checkpoint(input_size, hidden_layers, output_size,
                    model, p_drop, filename='checkpoint.pth'):
    "Save network architecture and parameters to file"
    checkpoint = {'input_size': input_size,
                  'hidden_layers': hidden_layers,
                  'output_size': output_size,
                  # only save state dict of classifier,
                  # not parameters layer of pre-trained network
                  'state_dict': model.classifier.state_dict(),
                  'p_drop': p_drop}

    torch.save(checkpoint, filename)


def process_image(image):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    im = Image.open(image)
    processing = transforms.Compose([
        transforms.Resize((256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return processing(im)
