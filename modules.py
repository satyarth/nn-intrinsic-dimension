import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

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
        self.proj = nn.Parameter(torch.randn(out_features*in_features, d), requires_grad=False)
        
        self.theta_0_b = nn.Parameter(torch.randn(out_features), requires_grad=False)
        self.proj_b = nn.Parameter(torch.randn(out_features, d), requires_grad=False)
        
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