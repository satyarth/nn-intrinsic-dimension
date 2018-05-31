import requests
from json import dumps, loads
from time import sleep

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from dispatcher import Dispatcher
from train_test import Dgrid_train
from models import ConvRP, FC_RP

tr_data = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
te_data = MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_data = torch.utils.data.DataLoader(dataset=tr_data, batch_size=64, shuffle=True)
test_data = torch.utils.data.DataLoader(dataset=te_data, batch_size=64, shuffle=True)

base_url = "http://10.30.103.213:5000/"

model_rp = ConvRP
opt = torch.optim.Adam
criterion = nn.CrossEntropyLoss()

while True:
    try:
        r = requests.get(base_url + "get_job")
        
    except requests.exceptions.ConnectionError:
        print("Can't reach server")
        sleep(5)
        continue
        
    response = loads(r.content)
    if response['status']:

        job_id = response['job_id']
        params = response['params']

        output = Dgrid_train(network_class=FC_RP, 
                            network_args={'d':None,
                                          'f_in': 28*28,
                                          'h1': 200,
                                          'h2': 200,
                                          'f_out': 10}, 
                            optimizer_class=opt,
                            optimizer_args={'lr':0.001},
                            criterion=criterion, 
                            train_data=train_data, 
                            test_data=test_data, 
                            epoches=30,
                            flatten = True,
                            d=params['d'], 
                            verbose=True)
        
        output["job_id"] = job_id

        requests.post(base_url + "post_results", data={'output': dumps(output)})

    else:
        sleep(5)
