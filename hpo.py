import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import json
import logging
import os
import sys

from io import BytesIO
from PIL import ImageFile
# to take care of the error: "image file is truncated (148 bytes not processed)"
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, validation_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in validation_loader:
            # use GPU if available           
            data=data.to(device)
            target=target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    average_loss = test_loss/len(validation_loader.dataset)
    average_accuracy = correct/len(validation_loader.dataset)

    logger.info(f"Test Average Loss: {average_loss}")
    logger.info(f"Test Average Accuracy: {average_accuracy}")


def train(model, train_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # limit the number of training images
    training_count = 0

    for epoch in range(1, epochs + 1):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # use GPU if available 
            data=data.to(device)
            target=target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            training_count += len(data)
            if training_count > 200: # for diagnosis, limit the count low.
                break
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item()
                    )
                )
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.required_grad = False

    num_features = model.fc.in_features
    num_classes = 133

    # add fully connected layer 
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 128),
                   nn.ReLU(),
                   nn.Linear(128, num_classes))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_path = os.path.join(data, 'train')
    validation_path = os.path.join(data, 'valid')

    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    test_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.ToTensor()
        ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)    
    validation_dataset = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)

    return train_data_loader, validation_data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model=model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, validation_loader = create_data_loaders(data=args.data_dir, batch_size=args.batch_size)
    model=train(model, train_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, validation_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    
    args=parser.parse_args()
    
    main(args)
