import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import constants
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):

        out = self.layer1(x)
        # out = self.layer2(out)
        # out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        # out = self.fc1(out)
        # out = self.fc2(out)
       #  return out
        return 0



'''

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(constants.N_MEL * constants.BLOCK_SIZE_BIN, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 88)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
