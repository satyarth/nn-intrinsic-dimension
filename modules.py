import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class LinearRP(nn.Module):
    # Commented stuff fron original linear layer for now
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, d, bias=True):
        super(LinearRP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.theta_0 = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        proj = torch.randn(out_features*in_features, d)
        proj = torch.div(proj,torch.norm(proj, p=2.,dim=0))
        self.proj = nn.Parameter(proj, requires_grad=False)
        
        self.theta_0_b = nn.Parameter(torch.randn(out_features), requires_grad=False)
        proj_b = torch.randn(out_features, d)
        proj_b = torch.div(proj_b,torch.norm(proj_b,p=2., dim=0))
        self.proj_b = nn.Parameter(proj_b, requires_grad=False)
            
        self.in_features = in_features
        self.out_features = out_features
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, basis_weights):
        offset_weight = torch.matmul(self.proj, basis_weights)
        offset_weight = offset_weight.resize(self.out_features, self.in_features) 
        offset_bias = torch.matmul(self.proj_b, basis_weights)
        offset_bias = offset_bias.resize(self.out_features) 
#         print(p1.resize(self.out_features, self.in_features).size())
        theta = self.theta_0 + offset_weight
        theta_b = self.theta_0_b + offset_bias
#         print(input.size(), theta.size(), theta_b.size())
        return F.linear(input, theta, theta_b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
class Conv2dRP(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1, 
                 bias=True, 
                 d=None):
        super(Conv2dRP, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.d = d
        
        #init filter weights & its projection matrix
        theta_0_w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.theta_0_w = nn.Parameter(theta_0_w, requires_grad=False)
        
        Proj_w = torch.rand(out_channels*in_channels*kernel_size*kernel_size, d)
        Proj_w = torch.div(Proj_w,torch.norm(Proj_w, p=2.,dim=0))
        self.Proj_w = nn.Parameter(Proj_w, requires_grad=False)
        
        #init filter biases & its projection matrix
        if bias:
            theta_0_b = torch.randn(out_channels)
            self.theta_0_b = nn.Parameter(theta_0_b, requires_grad=False)

            Proj_b = torch.rand(out_channels, d)
            Proj_b = torch.div(Proj_b,torch.norm(Proj_b, p=2.,dim=0))
            self.Proj_b = nn.Parameter(Proj_b, requires_grad=False)
        
    def forward(self, input, basis_weights):
        theta_off_w = torch.matmul(self.Proj_w, basis_weights)
        theta_off_w = theta_off_w.resize(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        Theta_w = self.theta_0_w + theta_off_w
        
        if self.bias: #if bias = True
            theta_off_b = torch.matmul(self.Proj_b, basis_weights)
            theta_off_b = theta_off_b.resize(self.out_channels)
            Theta_b = self.theta_0_b + theta_off_b
        else:
            Theta_b = None
            
        return F.conv2d(input, weight=Theta_w, bias=Theta_b, padding=self.padding, 
                        dilation=self.dilation, groups = self.groups)
    

class BatchNorm2dRP(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, d=None):
        super(BatchNorm2dRP, self).__init__()
        
        self.num_features=num_features
        self.eps=eps
        self.momentum=momentum
        self.affine=affine
        self.d = d
        
        if self.affine:
            #init filter weights & its projection matrix
            theta_0_w = torch.randn(self.num_features)
            self.theta_0_w = nn.Parameter(theta_0_w, requires_grad=False)

            Proj_w = torch.rand(self.num_features, d)
            Proj_w = torch.div(Proj_w,torch.norm(Proj_w, p=2.,dim=0))
            self.Proj_w = nn.Parameter(Proj_w, requires_grad=False)
            
            #init filter biases & its projection matrix
            theta_0_b = torch.randn(self.num_features)
            self.theta_0_b = nn.Parameter(theta_0_b, requires_grad=False)

            Proj_b = torch.rand(self.num_features, d)
            Proj_b = torch.div(Proj_b,torch.norm(Proj_b, p=2.,dim=0))
            self.Proj_b = nn.Parameter(Proj_b, requires_grad=False)

        self.running_mean=torch.ones(self.num_features) / self.num_features
        self.running_var=torch.zeros(self.num_features)
            
    def forward(self, input, basis_weights):
        
        if self.affine:
            theta_off_w = torch.matmul(self.Proj_w, basis_weights)
            theta_off_w = theta_off_w.squeeze()
            Theta_w = self.theta_0_w + theta_off_w
            
            theta_off_b = torch.matmul(self.Proj_b, basis_weights)
            theta_off_b = theta_off_b.squeeze()
            Theta_b = self.theta_0_b + theta_off_b
        else:
            Theta_w, Theta_b = 1, 0
        
        return F.batch_norm(input, self.running_mean, self.running_var, Theta_w, Theta_b,
            self.training, self.momentum, self.eps)