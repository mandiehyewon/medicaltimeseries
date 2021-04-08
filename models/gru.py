import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class GRU(nn.Module):
    def __init__(self,config):
        super(GRU,self).__init__()
        self.module_dict = nn.ModuleDict()

        self.config = config
        
        self.module_dict['rnn'] = nn.GRU(input_size=config.input_dim,
                                    hidden_size=100,
                                    num_layers=3,
                                    batch_first=True)

        # output_layer
        self.module_dict['output'] = nn.Linear(100,config.output_dim,bias=True)

    def forward(self, x):
        data = self.module_dict['rnn'](x)[0][:,-1,:]
        out = self.module_dict['output'](data)
        return F.log_softmax(out,1)

