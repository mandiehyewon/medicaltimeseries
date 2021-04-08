import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pdb
from datetime import datetime
import pickle
from tqdm import tqdm
import importlib
from config import config

from sklearn.metrics import roc_auc_score
from physionet import load_data

device = torch.device('cuda')

# create model
model_name = config.model_name
model_module = importlib.import_module("models." + model_name)
model_class = getattr(model_module, model_name.upper())

model = model_class(config).to(device)
model = nn.DataParallel(model)

train_x_numpy , train_y_numpy, test_x_numpy , test_y_numpy, config.input_dim = load_data(task=config.task)

# create training and test data
train_x = torch.from_numpy(train_x_numpy).type(torch.FloatTensor) 
train_y = torch.from_numpy(train_y_numpy).type(torch.FloatTensor)
test_x = torch.from_numpy(test_x_numpy).type(torch.FloatTensor) 
test_y = torch.from_numpy(test_y_numpy).type(torch.FloatTensor)


train_set = torch.utils.data.TensorDataset(train_x,train_y)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)

test_set = torch.utils.data.TensorDataset(test_x,test_y)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=True)


# start training
lr = config.LR
optimizer = optim.Adam(model.parameters(), lr = lr)

def main():
    num_epochs = config.NUM_EPOCHS
    
    if config.tensorboard:
        writer = SummaryWriter('logs_tb/'+config.model_name+'_'+str(datetime.now()))

    best_auc = 0
    best_epoch = 0
    try:
        for epoch_id in range(num_epochs):
            print('Epoch %d:'%(epoch_id))
            train_auc = train_epoch()
            test_auc = test_epoch()
            if test_auc > best_auc:
                best_auc = test_auc 
                best_epoch = epoch_id
                torch.save(model.state_dict(), 'saved/'+config.model_name)
            print('Best epoch %d: %f'%(best_epoch,best_auc))
            print()
            
            if config.tensorboard:
                writer.add_scalar('train_auc',train_auc,epoch_id)
                writer.add_scalar('test_auc',test_auc,epoch_id)
    
    except KeyboardInterrupt:
        pass
    finally:
        print(best_auc)
        if config.tensorboard:
            writer.close()


def train_epoch():
    total_loss = 0 
    correct = 0
    total_preds = []
    total_labels = []
    l2loss = nn.MSELoss(reduction='sum')
    for batch_id, [data,labels] in enumerate(tqdm(train_loader,ncols=75,leave=False)):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = F.binary_cross_entropy_with_logits(logits,labels)
        for param in model.parameters():
            loss += l2loss(param,torch.zeros_like(param)) * config.l2_coeff
        loss.backward()
        optimizer.step()

        preds = torch.sigmoid(logits).to("cpu").data.numpy()
        total_preds.append(preds)
        total_labels.append(labels.to("cpu").data.numpy())

    total_preds = np.concatenate(total_preds,0)
    total_labels = np.concatenate(total_labels,0)
    auc = roc_auc_score(total_labels,total_preds)
    print('>>> Train Auc: %.4f'%(auc))
    return auc


def test_epoch():
    total_loss = 0 
    correct = 0
    total_preds = []
    total_labels = []
    for batch_id, [data,labels] in enumerate(tqdm(test_loader,ncols=75,leave=False)):
        data, labels = data.to(device), labels.to(device)
        preds = 0
        for i in range(config.num_samples):
            logits = model(data)
            preds += torch.sigmoid(logits).to("cpu").data.numpy()

        preds = preds / config.num_samples
        total_preds.append(preds)
        total_labels.append(labels.to("cpu").data.numpy())
    total_preds = np.concatenate(total_preds,0)
    total_labels = np.concatenate(total_labels,0)
    auc = roc_auc_score(total_labels,total_preds)
    print('>>> Test Auc: %.4f'%(auc))
    return auc 


if __name__=='__main__':
    main()
