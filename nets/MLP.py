''' MLP model for the neural predicates in the DPL model'''
import torch
import torch.nn as nn

class MLP_inequality(nn.Module):
    def __init__(self,input_size,output_size):
        super(MLP_inequality,self).__init__()
        

        self.W1_1=nn.Linear(input_size,64)
        self.W1_2=nn.Linear(64,128)
        self.W1_3=nn.Linear(128,128)
        self.W1_4=nn.Linear(128,64)
        self.W1_5=nn.Linear(64,output_size)

        self.act=nn.Tanh()

        self.W2_1=nn.Linear(input_size,64)
        self.W2_2=nn.Linear(64,128)
        self.W2_3=nn.Linear(128,128)
        self.W2_4=nn.Linear(128,64)
        self.W2_5=nn.Linear(64,output_size)

        self.W3_1=nn.Linear(input_size,64)
        self.W3_2=nn.Linear(64,128)
        self.W3_3=nn.Linear(128,128)
        self.W3_4=nn.Linear(128,64)
        self.W3_5=nn.Linear(64,output_size)

    def forward(self,x):


        digit_1,digit_2,digit_3=torch.split(x,1,dim=1)

        digit_1=self.W1_1(digit_1)
        digit_1=self.act(digit_1)
        digit_1=self.W1_2(digit_1)
        digit_1=self.act(digit_1)
        digit_1=self.W1_3(digit_1)
        digit_1=self.act(digit_1)
        digit_1=self.W1_4(digit_1)
        digit_1=self.act(digit_1)
        digit_1=self.W1_5(digit_1)
        

        digit_2=self.W2_1(digit_2)
        digit_2=self.act(digit_2)
        digit_2=self.W2_2(digit_2)
        digit_2=self.act(digit_2)
        digit_2=self.W2_3(digit_2)
        digit_2=self.act(digit_2)
        digit_2=self.W2_4(digit_2)
        digit_2=self.act(digit_2)
        digit_2=self.W2_5(digit_2)
        

        digit_3=self.W3_1(digit_3)
        digit_3=self.act(digit_3)
        digit_3=self.W3_2(digit_3)
        digit_3=self.act(digit_3)
        digit_3=self.W3_3(digit_3)
        digit_3=self.act(digit_3)
        digit_3=self.W3_4(digit_3)
        digit_3=self.act(digit_3)
        digit_3=self.W3_5(digit_3)
        

        return digit_1,digit_2, digit_3
    
    