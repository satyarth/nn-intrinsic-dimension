import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from modules import Flatten, LinearRP, Conv2dRP, BatchNorm2dRP

class FC(nn.Module):
    def __init__(self, f_in, h1, h2, f_out):
        super(FC, self).__init__()
        
        self.h1 = nn.Linear(f_in, h1)
        self.h2 = nn.Linear(h1, h2)
        self.output = nn.Linear(h2, f_out)
        
        self.flat = Flatten()
        
        
    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.relu(x)
        return x

class FC_RP(nn.Module):
    def __init__(self, f_in, h1, h2, f_out, d):
        super(FC_RP, self).__init__()
        
        self.basis_weights = nn.Parameter(torch.zeros(d, 1))
        
        self.h1 = LinearRP(f_in, h1, d)
        self.h2 = LinearRP(h1, h2, d)
        self.output = LinearRP(h2, f_out, d)
        
        self.flat = Flatten()
        
        
    def forward(self, x):
        x = self.h1(x, self.basis_weights)
        x = F.relu(x)
        x = self.h2(x, self.basis_weights)
        x = F.relu(x)
        x = self.output(x, self.basis_weights)
        x = F.relu(x)
        return x
        
class ConvRP(nn.Module):
    def __init__(self, d):
        super(ConvRP, self).__init__()
        
        self.basis_weights = nn.Parameter(torch.zeros(d, 1))
        self.d = d
        
        self.conv1 = Conv2dRP(in_channels=1, out_channels=16, kernel_size=3, padding=1, d=d)
        self.bn1 = BatchNorm2dRP(num_features=16, d=d)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2) #14x14
        
        self.conv2 = Conv2dRP(in_channels=16, out_channels=32, kernel_size=3, padding=1, d=d)
        self.bn2 = BatchNorm2dRP(num_features=32, d=d)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(2) #7x7
        
        self.av = nn.AvgPool2d(7)
        self.lin1 = LinearRP(32, 10, d=d)
        
    def forward(self, x):
        
        x=self.conv1(x, self.basis_weights)
        x =self.bn1(x, self.basis_weights)
        x=self.relu1(x)
        x=self.mp1(x)
        x=self.conv2(x, self.basis_weights)
        x =self.bn2(x, self.basis_weights)
        x=self.relu2(x)
        x=self.mp2(x)
        
        x=self.av(x)
        x=self.lin1(x.view(x.size(0),-1), self.basis_weights)
        
        return x