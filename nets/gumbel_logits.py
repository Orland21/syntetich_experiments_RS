''' MLP used to obtain the logits of the permutations of the input data'''
import torch.nn as nn
import torch

class logits_permutations(nn.Module):

    def __init__(self,input_size,output_size):

        super(logits_permutations,self).__init__()
        self.W1=nn.Linear(input_size,32)
        self.W2=nn.Linear(32,16)
        self.W3=nn.Linear(16,output_size)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        x=self.W1(x)
        x=self.relu(x)
        x=self.W2(x)
        x=self.relu(x)
        x=self.W3(x)
        return x