import pandas as pd
import numpy as np

import torch
import torch.nn as nn

class NClassifier(nn.Module):
    def __init__(self):
        super(NClassifier, self).__init__()
        self.layer_1 = nn.Linear(66, 132) 
        self.layer_3 = nn.Linear(132, 66)
        self.layer_5 = nn.Linear(66, 10)
        self.layer_out = nn.Linear(10, 3) 
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.005)
        self.batchnorm = nn.BatchNorm1d(132)
        self.sig = nn.Sigmoid()
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm(x)
        x = self.tanh(self.layer_3(x))
        x = self.relu(self.layer_5(x))
        x = self.dropout(x)
        x = self.sig(self.layer_out(x))
        
        return x

def prep_ds(a):
        for j in range(0,len(a)):
            a[j] = np.concatenate(a[j])
        a = np.concatenate(a)
        return(a)

trained_model = NClassifier()
trained_model.load_state_dict(torch.load('saved_model.pth'))
device = torch.device("cpu")
trained_model.to(device)

def run(kp):
    global trained_model
    trained_model.eval()
    res = trained_model(torch.FloatTensor([kp]))
    return np.argmax((res).detach().numpy(),axis=1)[0]


