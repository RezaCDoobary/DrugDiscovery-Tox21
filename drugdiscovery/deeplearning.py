import torch
from torch.utils.data import TensorDataset
from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np

def get_data(X_train, y_train, mask_train, X_test, y_test, mask_test):
    train_set = TensorDataset(torch.tensor(X_train),torch.tensor(y_train.values),torch.tensor(mask_train.values))
    test_set = TensorDataset(torch.tensor(X_test),torch.tensor(y_test.values),torch.tensor(mask_test.values))
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=100, shuffle=True)
    return train_set, test_set, train_loader

def average_scores(y_pred_proba, y_true, score, weight):
    res = 0
    res= np.array([score(y_true[:,i],y_pred_proba[:,i], sample_weight = weight[:,i]) for i in range(0,12)])
    return res.mean()

def trainer(X,y,model,criterion, optimizer, assays, weights):
    model.zero_grad()  
    output = model(X) 
    criterion.weight = torch.tensor(np.array(weights)).float()
    loss = criterion(output, y)
    loss.backward()  
    optimizer.step()  
    return loss, output

def create_training_df(epochs):
    df = pd.DataFrame(columns = ['train loss','val loss','val auprc'],\
                index = [i+1 for i in range(epochs)])
    df.index = df.index.set_names('Epochs')
    return df

def train_model(model, model_name, optimizer, criterion, scheduler, epochs, train_loader, valid_test, \
                number_of_batches, assays, weight_train, weight_test, with_tqdm = False):
    """
    The main training function. 
    """
    
    df = create_training_df(epochs)
    
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()

        if with_tqdm:
            t = tqdm_notebook(enumerate(train_loader),total=number_of_batches, leave = True)
        else:
            t = enumerate(train_loader)
        for _,data in t:
            X,y, batch_weight = data[0].to(device).float() ,data[1].to(device).float(),data[2].to(device).long()
            tloss, output = trainer(X, y, model, criterion, optimizer, assays, batch_weight)
            epoch_loss += tloss.item()

        model.eval() 
        X_val, y_val, weight_test = test_set[:]
        preds = model(X_val.float())
        
            
        criterion.weight = torch.tensor(np.array(weight_test)).float()
        loss = criterion(preds, y_val.float())
        vloss = loss

        scheduler.step(vloss)

        train_loss = epoch_loss
        validation_loss = vloss
        
        np_preds = preds.clone().detach().numpy()
        auprc = average_scores(np_preds, y_val, average_precision_score, weight_test.numpy())
    
        print('Epoch [{}/{}] :: train loss: {:.3f} val loss: {:.3f} - val AUPRC: {:.3f}'
            .format(epoch+1,epochs,train_loss, validation_loss, auprc))
        df.loc[epoch+1] = [train_loss, validation_loss.item(), auprc]
    return df

import torch
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class net(nn.Module):
    def __init__(self, input_size, output_size, seed, hidden_layers  = [64,64], \
                 activations = [torch.relu,torch.relu],drop_p = 0.3):
        super(net, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activations = activations
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)
        self.bat = [nn.BatchNorm1d(hidden_layers[i]) for i in range(0,len(hidden_layers))]

    def forward(self, input):
        for i,linear in enumerate(self.hidden_layers):
            input = self.activations[i](linear(input))
       
            input = self.bat[i](input)
            input = self.dropout(input)
        
        output = self.output(input)
        output = torch.sigmoid(output)
        return output
