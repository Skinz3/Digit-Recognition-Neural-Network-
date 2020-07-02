import librosa
import soundfile as sf
import librosa.display

import torch
import torchvision
from torch import nn
from torchvision import transforms, datasets
import numpy
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as image
import sys
from model import Net
from dataset import MapsDataset
import constants


def train():

    train = MapsDataset("MAPS/flac/","MAPS/tsv/")

    test = MapsDataset("MAPS/flac_test/","MAPS/tsv_test/")

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

    print("Starting training...")

    net = Net()

    # Too few epochs, and your model wont learn everything it could have.
    # Too many epochs and your model will over fit to your in-sample data 
    EPOCHS = 3
    # learning rate dictate the magnitude of changes optimizer can make at a time,
    # to high learning rate can lead to chaotic gradiant descent
    LEARNING_RATE = 0.001

    

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE) 

    for epoch in range(EPOCHS): # We iterate 3 time over all our training data data
        for i,data in enumerate(trainset):  # iterate over training data

            # X is cnn input data (a window of spectrogram)
            # y is expected result (a note)
            X, y = data  

            net.zero_grad()  # sets gradients to 0 before loss calc every step

            output = net(X.view(-1,constants.N_MEL * constants.BLOCK_SIZE_BIN))  # pass in the reshaped batch (Width = 100 (window size) Height = 229 (log scale))

            loss = F.nll_loss(output, y)  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights 
        print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines! 


    correct = 0
    total = 0

    with torch.no_grad():
        for data in trainset:
            X, y = data # X is i cropped spectrogram. y is the midi note

            output = net(X.view(-1,constants.N_MEL * constants.BLOCK_SIZE_BIN))

            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]: # Has the network planned the right note ?
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))  

    torch.save(net.state_dict(), "model.pt")



def evaluate():
    net = Net() 
    net.load_state_dict(torch.load("model.pt"))
    net.eval()

    im = Image.open("test.png") # Can be many different formats.
    pixels =  torch.FloatTensor((list(im.getdata())))
    
    result = net(pixels.view(-1,28*28))

    print(torch.argmax(result))
    


train()

 

