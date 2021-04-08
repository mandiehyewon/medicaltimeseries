import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class UA(nn.Module):
    def __init__(self,config):
        super(UA,self).__init__()
        self.module_dict = nn.ModuleDict()

        self.config = config
        self.module_dict['embed'] = nn.Linear(config.input_dim,config.hidden_dim,False)
        self.module_dict['rnn_alpha'] = nn.LSTM(input_size=config.hidden_dim,
                                    hidden_size=config.hidden_dim,
                                    num_layers=config.num_layers,
                                    batch_first=True,
                                    dropout=0.75)
        self.module_dict['rnn_beta'] = nn.LSTM(input_size=config.hidden_dim,
                                    hidden_size=config.hidden_dim,
                                    num_layers=config.num_layers,
                                    batch_first=True,
                                    dropout=0.75)
        self.module_dict['alpha_mu'] = nn.Linear(config.hidden_dim,1,bias=True)
        self.module_dict['alpha_sigma'] = nn.Linear(config.hidden_dim,1,bias=True)
        self.module_dict['beta_mu'] = nn.Linear(config.hidden_dim,config.hidden_dim,bias=True)
        self.module_dict['beta_sigma'] = nn.Linear(config.hidden_dim,config.hidden_dim,bias=True)      

        # output_layer
        self.module_dict['output'] = nn.Linear(config.hidden_dim,config.output_dim,bias=True)

    def forward(self, x):
        embed = self.module_dict['embed'](x)
        reversed_embed = torch.flip(embed, [1])

        alpha_outputs = self.module_dict['rnn_alpha'](reversed_embed)[0]
        beta_outputs = self.module_dict['rnn_beta'](reversed_embed)[0]

        # alpha_att
        alpha_mu = self.module_dict['alpha_mu'](alpha_outputs)
        alpha_sigma = self.module_dict['alpha_sigma'](alpha_outputs) 
        alpha_sigma = F.softplus(alpha_sigma)
        alpha_dist = torch.distributions.normal.Normal(loc=alpha_mu, scale=alpha_sigma)
        alpha_att = alpha_dist.sample()
        alpha_att = torch.softmax(alpha_att,1)
        reversed_alpha_att = torch.flip(alpha_att,[1])       

        # beta_att
        beta_mu = self.module_dict['beta_mu'](beta_outputs)
        beta_sigma = self.module_dict['beta_sigma'](beta_outputs)
        beta_sigma = F.softplus(beta_sigma)
        beta_dist = torch.distributions.normal.Normal(loc=beta_mu, scale=beta_sigma)
        beta_att = beta_dist.sample()
        beta_att = torch.tanh(beta_att)
        reversed_beta_att = torch.flip(beta_att,[1])
        
        c_i = torch.sum(embed * reversed_alpha_att * reversed_beta_att, 1)

        out = self.module_dict['output'](c_i)
        return out

