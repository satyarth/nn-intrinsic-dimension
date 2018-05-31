import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from modules import Flatten, LinearRP, Conv2dRP, BatchNorm2dRP

class FC(nn.Module):
    def __init__(self, f_in, h1, h2, f_out):
        super(FC, self).__init__()
        
        self.h1 = nn.Linear(f_in, h1)
        self.relu1 = nn.ReLU(inplace=True)
        self.h2 = nn.Linear(h1, h2)
        self.relu2 = nn.ReLU(inplace=True)
        self.output = nn.Linear(h2, f_out)
        
        self.flat = Flatten()
        
        
    def forward(self, x):
        x = self.relu1(self.h1(x))
        x = self.relu2(self.h2(x))
        x = self.output(x)
        return x

class FC_RP(nn.Module):
    def __init__(self, f_in, h1, h2, f_out, d):
        super(FC_RP, self).__init__()
        self.d = d
        self.basis_weights = nn.Parameter(torch.zeros(d, 1))
        
        self.h1 = LinearRP(f_in, h1, d)
        self.relu1 = nn.ReLU(inplace=True)
        self.h2 = LinearRP(h1, h2, d)
        self.relu2 = nn.ReLU(inplace=True)
        self.output = LinearRP(h2, f_out, d)
        
        self.flat = Flatten()
        
        
    def forward(self, x):
        #x = self.flat(x)
        x = self.relu1(self.h1(x, self.basis_weights))
        x = self.relu2(self.h2(x, self.basis_weights))
        x = self.output(x, self.basis_weights)
        return x

class ConvRP(nn.Module):
    def __init__(self, d):
        super(ConvRP, self).__init__()
        
        self.basis_weights = nn.Parameter(torch.zeros(d, 1))
        self.d = d
        
        self.conv1 = Conv2dRP(in_channels=1, out_channels=16, kernel_size=3, padding=1, d=d)
        #self.bn1 = BatchNorm2dRP(num_features=16, d=d)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2) #14x14
        
        self.conv2 = Conv2dRP(in_channels=16, out_channels=32, kernel_size=3, padding=1, d=d)
        #self.bn2 = BatchNorm2dRP(num_features=32, d=d)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(2) #7x7
        
        self.av = nn.AvgPool2d(7)
        self.lin1 = LinearRP(32, 10, d=d)
        
    def forward(self, x):
        
        x=self.conv1(x, self.basis_weights)
        #x =self.bn1(x, self.basis_weights)
        x=self.relu1(x)
        x=self.mp1(x)
        x=self.conv2(x, self.basis_weights)
        #x =self.bn2(x, self.basis_weights)
        x=self.relu2(x)
        x=self.mp2(x)
        
        x=self.av(x)
        x=self.lin1(x.view(x.size(0),-1), self.basis_weights)
        
        return x

    
class ConvNet(nn.Module):
    def __init__(self, c_in=1, c1=50, c2=100):
        super(ConvNet, self).__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c1, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(2)
        
        self.av = nn.AvgPool2d(5)
        self.lin1 = nn.Linear(c2, 10)
        
    def forward(self, x):
        
        x=self.relu1(self.conv1(x))
        x=self.mp1(x)
        
        x=self.relu2(self.conv2(x))
        x=self.mp2(x)
        
        x=self.av(x)
        x=self.lin1(x.view(x.size(0),-1))
        
        return x
    
    
class ConvNetRP(nn.Module):
    def __init__(self, d, c_in=1, c1=50, c2=100):
        super(ConvNetRP, self).__init__()
        
        self.basis_weights = nn.Parameter(torch.zeros(d, 1))
        self.d = d
        
        self.conv1 = Conv2dRP(in_channels=c_in, out_channels=c1, kernel_size=3, padding=0, d=d)
        self.relu1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(2)
        
        self.conv2 = Conv2dRP(in_channels=c1, out_channels=c2, kernel_size=3, padding=0, d=d)
        self.relu2 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(2)
        
        self.av = nn.AvgPool2d(5)
        self.lin1 = LinearRP(c2, 10, d=d)
        
    def forward(self, x):
        
        x = self.relu1(self.conv1(x, self.basis_weights))
        x = self.mp1(x)
        x = self.relu2(self.conv2(x, self.basis_weights))
        x=self.mp2(x)
        
        x=self.av(x)
        x=self.lin1(x.view(x.size(0),-1), self.basis_weights)
      
        return x
    
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(2) 
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(2) 
        
        self.lin1 = nn.Linear(120, 10)
        self.relu3 = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(84, 10)
        
    def forward(self, x):
        
        x=self.relu1(self.conv1(x))
        x=self.mp1(x)
                     
        x=self.relu2(self.conv2(x))
        x=self.mp2(x)

        x=self.relu3(self.lin1(x.view(x.size(0),-1)))
        x=self.lin2(x)
        
<<<<<<< HEAD
        return x
=======
        return x
>>>>>>> 2b91aac90ff0adf83ff444ad162c38d7149da2fd
