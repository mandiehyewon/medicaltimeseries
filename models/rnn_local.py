import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.nn import Parameter

from torch.nn import init 



class RNN_LOCAL(nn.Module):
    def __init__(self,config):
        super(RNN_LOCAL,self).__init__()
        self.module_dict = nn.ModuleDict()

        self.config = config
        
        self.module_dict['rnn'] = CUSTOM_RNN(input_size=config.input_dim,
                                    hidden_size=config.hidden_dim, num_layers=1)

        # output_layer
        self.module_dict['output'] = nn.Linear(config.hidden_dim,config.output_dim,bias=True)

    def forward(self, x):
        data = self.module_dict['rnn'](x)
        out = self.module_dict['output'](data)
        return out




class CUSTOM_RNN_LAYER(nn.Module):
    def __init__(self, input_size, hidden_size,
                bias=True, dropout=0.):
        super(CUSTOM_RNN_LAYER,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout

        self.layer_cin = nn.Linear(input_size,hidden_size,bias=True)
        self.layer_c = nn.Sequential(
                            nn.Linear(2*hidden_size,2*hidden_size),
                            nn.ReLU(),
                            nn.Linear(2*hidden_size,2*hidden_size),
                            nn.ReLU(),
                            nn.Linear(2*hidden_size,2*hidden_size),
                            nn.ReLU(),
                            nn.Linear(2*hidden_size,1))

        self.threshold1 = nn.Threshold(0.5,0)
        self.threshold2 = nn.Threshold(-0.4,1)

        self.gru = nn.GRUCell(hidden_size,hidden_size)


    def forward(self, input, h0=None):
        c = torch.zeros(input.shape[0], self.hidden_size,
                             dtype=input.dtype, device=input.device)
        if h0 is None:
            h = torch.zeros(input.shape[0], self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            h = h0
        n = torch.zeros(input.shape[0],1,dtype=input.dtype, device=input.device)
        
        c_concat = []
        h_concat = []
        for s in range(input.shape[1]):
            c_in = torch.tanh(self.layer_cin(input[:,s,:]))
            com = torch.cat([c,c_in],1)

            # alpha = torch.sigmoid(self.layer_c(com))
            # alpha = self.threshold1(alpha)
            # alpha = self.threshold2(-alpha)

            alpha_logit = self.layer_c(com)
            ber_dist = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=1e-1,logits=alpha_logit)
            alpha = ber_dist.sample()

            h = h*(1-alpha) + alpha * self.gru(c,h)

            new_n = n * (1-alpha) + 1
            c = (c * n * (1-alpha) + c_in)/ new_n
            n  = new_n
            
            c_concat.append(c_in.view(-1,1,self.hidden_size))
            h_concat.append(h.view(-1,1,self.hidden_size))

        h = self.gru(c,h)
        c_concat = torch.cat(c_concat,1)
        h_concat = torch.cat(h_concat,1)
        return c_concat, h_concat, h

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

class CUSTOM_RNN(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers, bias=True, dropout=0.):
        super(CUSTOM_RNN,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.num_layers = num_layers

        self.module_dict = nn.ModuleDict()
        self.module_dict["0"] = CUSTOM_RNN_LAYER(input_size, hidden_size, bias, dropout)
        for i in range(1,num_layers):
            self.module_dict[str(i)] = CUSTOM_RNN_LAYER(2*hidden_size, hidden_size, bias, dropout)
    
    def forward(self,input):
        data = input
        h = None
        for i in range(self.num_layers):
            c_concat, h_concat, h  = self.module_dict[str(i)](data,h)
            data = torch.cat([c_concat,h_concat],2)
        return h
