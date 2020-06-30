################
# Author: Reza C Doobary
#
# deeplearning.py
#
# This script provides all deep neural network functionalities pertinent to the training of the models.
################

# Imports
import torch
from torch.utils.data import TensorDataset
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from collections import deque
import torch
import torch.nn.functional as F
import torch.nn as nn


def get_data(X_train:np.array, y_train:np.array, mask_train:np.array, X_test:np.array, \
    y_test:np.array, mask_test:np.array, batch_size:np.array)->(torch.utils.data.Dataset,torch.utils.data.Dataset,torch.utils.data.DataLoader,int):
    """
    Data preparation of the training and testing data into torch.utils.data.Dataset torch.utils.data.DataLoader objects to be consumed by
    a pytorch training routine.
    """
    train_set = TensorDataset(torch.tensor(X_train),torch.tensor(y_train.values),torch.tensor(mask_train.values))
    test_set = TensorDataset(torch.tensor(X_test),torch.tensor(y_test.values),torch.tensor(mask_test.values))
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    number_of_batches = int(len(train_set)/batch_size)
    return train_set, test_set, train_loader, number_of_batches



class net(nn.Module):
    """
    The neural network used in this project.

    This is a fairly customisation model with the addition of resnet layers for regularity. 

    Basic usage:

    >> layers = [1024,2048,4196]
    >> activations = [torch.relu]*len(layers)
    >> model = net(2048, 12, seed = 12345, hidden_layers = layers, activations = activations).to(device)

    """
    def __init__(self, input_size:int, output_size:int, seed:int, hidden_layers:list  = [64,64], \
                 activations:list = [torch.relu,torch.relu],drop_p:float = 0.5, res_layers:bool = True):
        super(net, self).__init__()

        # configuration features:
        self.res_layers = res_layers # If true - will have residual layers
        self.seed = torch.manual_seed(seed)
        self.dropout = nn.Dropout(p=drop_p)
        self.activations = activations # Activations

        # Implementing hidden layers:     
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])]) # first layer, from input to first hidden layer
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:]) 
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes]) # hidden layers implemented here
        self.constant_layers = nn.ModuleList([nn.Linear(h,h) for h in hidden_layers]) # constant layers are implemented here (for residual layers)
        self.output = nn.Linear(hidden_layers[-1], output_size) # the output layer

        # batch normalisation layers implemented here:
        self.bat = nn.ModuleList([nn.BatchNorm1d(hidden_layers[i]) for i in range(0,len(hidden_layers))])

    def forward(self, input):
        # step 1. For each hidden layer:
        for i,linear in enumerate(self.hidden_layers):
            # step 2. Apply activation after layer
            input = self.activations[i](linear(input))

            # step 3. If residual layers are enabled: add another constant layer from the previous layer output, with an activation
            if self.res_layers:
              input = self.activations[i](self.constant_layers[i](input)) + input
       
            # step 4. Batch normalise
            input = self.bat[i](input)

            # step 5. Dropout
            input = self.dropout(input)

        # step 6. output layer and sigmoid    
        output = self.output(input)
        output = torch.sigmoid(output)
        return output

class EarlyStopping:
    """
    Early stopping class used for regulating training if the training does not progress after a given patience.

    Basic usage:

    >> early_stopper = EarlyStopping(patience=10)
    >> if early_stopper(12.2):
    >>    break
    """
    def __init__(self, patience:int):
        self.patience = patience
        self.values = deque(maxlen = self.patience) # This is used to control the patience of the stopper.
        self.current_max = -np.inf
        
    def should_break(self, value:float):
        # step 1. Update max
        self.current_max = max(self.current_max, value) 

        # step 2. Add values to the deque
        self.values.append(value)

        # step 3. If the current max is larger than all item in the deque then break.
        if len(self.values) == self.patience and ((np.array(self.values) < self.current_max) == True).all():
            return True
        
        return False

class Trainer:
    """
    The main trainer class responsible to training a pytorch model given the optimiser, criterion, scheduler, and early stoppage configurations.

    Basic usage:

    >> activations = [torch.relu]*len(layers)
    >> early_stopper = EarlyStopping(patience=10)
    >> model = dl.net(2048, 12, seed = 12345, hidden_layers = layers, activations = activations).to(device)
    >> optimizer = torch.optim.Adam(model.parameters(), lr=4e-5,weight_decay= 1e-5)
    >> criterion = nn.BCELoss()
    >> scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = False)
    >> training_module = Trainer(model, optimizer,criterion,epochs, root, device, scheduler ,early_stopper)
    >> model_name = 'config_N'+ str(config_n) +'_'+'_'.join([str(l) for l in layers])
    >> results = training_module.train_model(train_loader, test_set, number_of_batches, targets, mask_train, mask_test, model_name)
    """
    def __init__(self, model, optimiser, criterion, epochs, root, device, scheduler = None, early_stopper = None):
        # note that type declaration has been left empty since the pytorch types are specified for their use case.
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.epochs = epochs
        self.scheduler = scheduler
        self.early_stopper = early_stopper
        self.root = root
        self.device = device

    def _average_scores(self, y_pred_proba:np.array, y_true:np.array, score, weight:np.array)->float:
        """
        Return the 'score' for the y_pred_proba and y_true, where the score is any function providing a metric for these appropriate entries.
        """
        res = np.array([score(y_true[:,i],y_pred_proba[:,i], sample_weight = weight[:,i]) for i in range(0,12)])
        return res.mean()

    def _training_routine(self, X:np.array, y:np.array, assays:np.array, weights:np.array):
        """
        The main training routine which takes the data input X, the observation y, and the weights (which mask null values in the observation data),
        and performs one training set whilst returning the loss and output of the model.
        """
        self.model.zero_grad()  
        output = self.model(X).to(self.device)
        self.criterion.weight = torch.tensor(weights).to(self.device).float()
        loss = self.criterion(output, y).to(self.device)
        loss.backward()  
        self.optimiser.step()  
        return loss, output

    def _create_training_df(self)->pd.core.frame.DataFrame:
        df = pd.DataFrame(columns = ['train loss','val loss','val auprc'],\
                    index = [i+1 for i in range(self.epochs)])
        df.index = df.index.set_names('Epochs')
        return df

    def train_model(self, train_loader, valid_test, \
                    number_of_batches, assays, weight_train, weight_test, model_name, with_tqdm = False,\
                    print_every = 10):
        """
        The main training function. 
        """
        minimum_val_loss = float('inf')
        df = self._create_training_df()

        # step 1. Beginning training for a given number of epochs.
        for epoch in range(self.epochs):
            epoch_loss = 0
            self.model.train()

            # step 2. Enumerate over the number of items in train_loader
            if with_tqdm:
                t = tqdm_notebook(enumerate(train_loader),total=number_of_batches, leave = True)
            else:
                t = enumerate(train_loader)

            # step 3. Pull a batch from the train loader
            for _,data in t:
                X,y, batch_weight = data[0].to(self.device).float() ,data[1].to(self.device).float(),data[2].to(self.device).long()

                # step 4. Apply the training routine.
                tloss, output = self._training_routine(X, y, assays, batch_weight)
                epoch_loss += tloss.item()

            # step 4. Prepare to evaluate on the test set, and make a prediction for the validation set.
            self.model.eval() 
            X_val, y_val, weight_test = valid_test[:]
            preds = self.model(X_val.to(self.device).float()).to(self.device)
            
            # step 5. Re-adjust the criterion weight to mask null values in the validation target set.
            self.criterion.weight = torch.tensor(weight_test).to(self.device).float()
            loss = self.criterion(preds.to(self.device), y_val.to(self.device).float()).to(self.device)
            vloss = loss

            # step 6. Apply scheduler if vloss is not progressing enough.
            if self.scheduler is not None:
                self.scheduler.step(vloss)

            train_loss = epoch_loss
            validation_loss = vloss

            # step 7. Compute AUPRC and AUCROC.
            np_preds = preds.clone().detach().cpu().numpy()
            auprc = self._average_scores(np_preds, y_val, average_precision_score, weight_test.numpy())
            aucroc = self._average_scores(np_preds, y_val, roc_auc_score, weight_test.numpy())

            if epoch%print_every == 0:
            print('Epoch [{}/{}] :: train loss: {:.6f} val loss: {:.6f} - val AUPRC: {:.3f} - val AUCROC :{:.3f}' 
                .format(epoch,self.epochs,train_loss, validation_loss, auprc, aucroc))

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimiser.state_dict()},
                self.root + 'models/dnn_models/' + model_name + '.pth')

           
            # step 8. If early stop - then stop else print the current training loss, validation loss, and current AUPRC.
            if self.early_stopper:
                if self.early_stopper.should_break(auprc):
                print('Breaking due to lack of progress in the AUPRC every ' + str(self.early_stopper.patience))
                return df

            else:
            print('\r', 'Epoch [{}/{}]'.format(epoch,self.epochs), end='')
            df.loc[epoch+1] = [train_loss, validation_loss.item(), auprc]
        return df



# This function is used for NLP investigation and is subject to change.
def get_data2(X_train, y_train, mask_train, X_test, y_test, mask_test, batch_size):
    train_set = TensorDataset(torch.tensor(X_train),torch.tensor(y_train.values),torch.tensor(mask_train.values))
    test_set = TensorDataset(torch.tensor(X_test),torch.tensor(y_test.values),torch.tensor(mask_test.values))
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    return train_set, test_set, train_loader, test_loader




    