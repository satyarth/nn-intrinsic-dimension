import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from modules import Flatten, LinearRP

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
        