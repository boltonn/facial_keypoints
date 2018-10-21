## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## output size = (W-F)/S +1 = (224-4)/1 +1 = 221
        ## output size = (W-F)/S +1 = (96-4)/1 +1 = 93 (using 96x96 images as suggested in Naimish paper)
        #self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv1 = nn.Conv2d(1, 32, 5) #trying to reduce number of parameters to speed up training        
        #nn.init.uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(p=0.1)
        # (32, 110, 110); 110.5 is rounded down
        # (32, 46, 46); 46.5 is rounded down
        
        #self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2 = nn.Conv2d(32, 64, 5)
        #nn.init.uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(p=0.2)
        # (64, 54, 54)
        # (64, 22, 22)
        
        #self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv3 = nn.Conv2d(64, 128, 5)
        #nn.init.uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(p=0.3)
        # (128, 26, 26); 26.5 is rounded down
        # (128, 10, 10)
        
        #self.conv4 = nn.Conv2d(128, 256, 1)
        self.conv4 = nn.Conv2d(128, 256, 5)
        #nn.init.uniform_(self.conv4.weight)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout(p=0.4)
        # (256, 13, 13)
        # (256, 5, 5)
        #self.avgpool = nn.AvgPool2d(13, stride=1)
        
        
        #self.fc1 = nn.Linear(256*13*13, 1000)
        self.fc1 = nn.Linear(256*10*10, 2048) #increase filter size and nodes in dense layer
        #self.fc1 = nn.Linear(256*5*5, 1000) #(96x96 pixel input)
        #self.fc1 = nn.Linear(128*10*10, 1000) #(96x96 pixel input, no 4th layer)
        #self.fc1 = nn.Linear(256, 256) #applied global average pooling
        nn.init.xavier_uniform_(self.fc1.weight)
        self.bn5 = nn.BatchNorm1d(2048)
        self.drop5 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(2048, 1024)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.bn6 = nn.BatchNorm1d(1024)
        self.drop6 = nn.Dropout(p=0.6)
        
        self.fc3 = nn.Linear(1024, 136)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        self.pool = nn.MaxPool2d(2,2)

        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        x = self.drop1(self.pool(self.bn1(F.elu(self.conv1(x)))))
        x = self.drop2(self.pool(self.bn2(F.elu(self.conv2(x)))))
        x = self.drop3(self.pool(self.bn3(F.elu(self.conv3(x)))))
        x = self.drop4(self.pool(self.bn4(F.elu(self.conv4(x)))))        
        x = x.view(x.size(0), -1)
        x = self.drop5(self.bn5(F.elu(self.fc1(x))))
        x = self.drop6(self.bn6(F.elu(self.fc2(x))))
        x = self.fc3(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.drop5(self.bn5(F.elu(self.fc1(x))))
        #x = self.fc3(x)
        
        return x
