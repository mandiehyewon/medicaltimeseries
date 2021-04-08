import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class LSTM(nn.Module):
    def __init__(self,config):
        super(LSTM,self).__init__()
        self.module_dict = nn.ModuleDict()

        self.config = config
        
        self.module_dict['rnn'] = nn.LSTM(input_size=config.input_dim,
                                    hidden_size=config.hidden_dim,
                                    num_layers=config.num_layers,
                                    batch_first=True,
                                    dropout=config.dropout)

        # output_layer
        self.module_dict['output'] = nn.Linear(config.hidden_dim,config.output_dim,bias=True)

    def forward(self, x):
        data = self.module_dict['rnn'](x)[0][:,-1,:]
        out = self.module_dict['output'](data)
        return out

