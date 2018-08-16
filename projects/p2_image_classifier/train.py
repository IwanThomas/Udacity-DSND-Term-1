import numpy as np
import argparse
import sys
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import utility_functions as h


class Network(nn.Module):
    "Class for neural network"
    def __init__(self, input_size, hidden_layers, output_size, p_drop):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size,
                                                      hidden_layers[0])])

        # add arbitrary number of hidden layers
        layers = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layers])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        output = self.output(x)

        return F.log_softmax(output, dim=1)


def create_model(arch, input_size, hidden_layers,output_size, p_drop):
    "Downloads pretrained model and returns untrained model"
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print ("Please choose one of vgg16 or vgg19. The script will now terminate")
        sys.exit()
        
    print ('Initiating Classifier')
    classifier = Network(input_size, hidden_layers, output_size, p_drop)

    # replace the pre-trained classifier
    print ('Replacing pre-trained classifier with problem-specific one')
    model.classifier = classifier
    return model


def train_model(model, trainloader, validateloader, device,
                learning_rate, epochs):
    "Trains the neural network"
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    for param in model.features:
        param.requires_grad = False

    # move device to CPU or GPU
    model.to(device)

    epochs = epochs
    print_every = 40
    steps = 0
    print ("Beginning training")
    for e in range(epochs):
        running_loss = 0
        for data, target in trainloader:
            steps += 1

            # Move input and label tensors to the relevant device
            data, target = data.to(device), target.to(device)

            # set optimizer step to zero to prevent it from accumulating
            optimizer.zero_grad()

            # perform a forward and backward pass
            output = model.forward(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # set model to eval mode to compute accuracy
                model.eval()
                train_accuracy = calculate_accuracy(model, trainloader, device)
                validation_accuracy = calculate_accuracy(model, validateloader, device)

                print("Epoch {}/{}".format(e+1, epochs),
                      "Training Loss {:.4f}".format(running_loss/print_every),
                      "Training Accuracy {:.4f} %".format(train_accuracy*100),
                      "Validation Accuracy {:.4f} %".format(validation_accuracy*100))

                # reset running loss to zero
                running_loss = 0

                # set model back to train mode
                model.train()

    # return trained model
    return model

def calculate_accuracy(model, dataloader, device):
    correct = 0
    total = 0

    # turn off gradients to save computation
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target.data).sum().item()

    return (correct / total)


def main(data_dir, learning_rate, epochs, device, arch):
    "Pull everything together"

    input_size=25088
    hidden_layers=[4096, 1024]
    output_size=102
    p_drop=0.5
    model = create_model(arch, input_size, hidden_layers, output_size, p_drop)

    trainloader, validateloader, _ = h.load_data(data_dir)
    model = train_model(model, trainloader, validateloader, device,
                        learning_rate=learning_rate, epochs=epochs)
    # save checkpoint
    h.save_checkpoint(input_size=input_size, hidden_layers=hidden_layers,
                      output_size=output_size,model=model, p_drop=p_drop,
                      filename='checkpoint.pth')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('data_dir',
                        type=str,
                        help='Path to input data directory')
    parser.add_argument('--learning_rate',
                        type=float,
                        help='Learning rate used when training model')
    parser.add_argument('--epochs',
                        type=int,
                        help='Training epochs used when training model')
    parser.add_argument('--gpu',
                        action='store_true',
                        help='Flag indicating whether GPU should be used for training')
    parser.add_argument('--arch',
                        type=str,
                        help='Choose either the vgg19 or vgg16 architecture \
                             the pre-trained model')
    
    args = parser.parse_args()
    data_dir = args.data_dir
    learning_rate = args.learning_rate if args.learning_rate else 0.001
    epochs = args.epochs if args.epochs else 1
    device = "cuda:0" if args.gpu == 'yes' and torch.cuda.is_available() else "cpu"
    arch = args.arch if args.arch else 'vgg19'
    
    main(data_dir, learning_rate, epochs, device, arch)
    