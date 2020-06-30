################
# Author: Reza C Doobary
#
# dataintegrity.py
#
# The purpose of the script is provide a testing framework functionality for the purposes of computing the following metrics for all of the assays:
#                           ['Precision','Recall', 'F1', 'AUPRC', 'Accuracy','Balanced Accuracy','ROC_AUC']
#   as well as creating 12 precision-recall plots for the different targets.
#
################

# Imports
import numpy as np 
import pandas as pd 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score,average_precision_score
from tqdm import tqdm_notebook, tqdm
import matplotlib.pyplot as plt

class panel_of_test(object):
    """
    Class responsible for the panel of tests according to 'Precision','Recall', 'F1', 'AUPRC', 'Accuracy','Balanced Accuracy','ROC_AUC', 
    and provides functionality for creating precision-recall curves, (subject to null weightings).

    Basic usage:
    >> targets = ['SR-HSE','NR-AR', 'SR-ARE', 'NR-Aromatase', 'NR-ER-LBD', 'NR-AhR', 'SR-MMP',\
       'NR-ER', 'NR-PPAR-gamma', 'SR-p53', 'SR-ATAD5', 'NR-AR-LBD']
    >> test_panel = panel_of_test(targets, X_test, y_test, sample_weights = mask_test)
    >> y_score = model(torch.tensor(X_test).cuda().float()).cpu().detach().numpy()
    >> y_pred = np.array(y_score>0.5)
    >> metrics = test_panel.compute_basic_metrics(y_pred, y_score)

     To produce the precision-recall plots do:
    >>test_panel.plot_recall_precision(y_score, extra_index = False)
    
    """
    def __init__(self, assays:np.array, X:np.array, y:np.array, notebook:bool = True, sample_weights:np.array = None):
        self.assays = assays
        self.X = X
        self.y = y
        self.notebook = notebook
        self.sample_weights = sample_weights 
        
    def compute_basic_metrics(self, y_pred:np.array, y_score:np.array)->pd.core.frame.DataFrame:
        """
        Computes the 'Precision','Recall', 'F1', 'AUPRC', 'Accuracy','Balanced Accuracy','ROC_AUC' for the predicted y's, and returns the results as
        as dataframe. 
        """
        
        result = pd.DataFrame(index = self.assays, columns = ['Precision','Recall', 'F1', 'AUPRC', 'Accuracy','Balanced Accuracy','ROC_AUC'])

        if self.notebook:
            t = tqdm_notebook(range(0,len(self.assays)), leave = True)
        else:
            t = tqdm(range(0,len(self.assays)), leave = True)

        for i in t:
            precision = precision_score(self.y[:,i],y_pred[:,i], sample_weight = self.sample_weights[:,i])
            recall = recall_score(self.y[:,i],y_pred[:,i],sample_weight = self.sample_weights[:,i])
            f1 = f1_score(self.y[:,i], y_pred[:,i],sample_weight = self.sample_weights[:,i])
            auprc = average_precision_score(self.y[:,i],y_score[:,i], sample_weight = self.sample_weights[:,i])
            acc = accuracy_score(self.y[:,i], y_pred[:,i],sample_weight = self.sample_weights[:,i])
            bal_acc = balanced_accuracy_score(self.y[:,i], y_pred[:,i],sample_weight = self.sample_weights[:,i])
            aucscore = roc_auc_score(self.y[:,i], y_pred[:,i],sample_weight = self.sample_weights[:,i])

            result.loc[self.assays[i]] = [precision, recall, f1, auprc ,acc, bal_acc, aucscore]
        return result
    
    def _plot_precision_recall_curve(self, y_test, y_pred_proba, assay_idx, plot_grid_index_tuple, axs, assays):
        
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

        i,j = plot_grid_index_tuple[0], plot_grid_index_tuple[1]
        axs[i][j].plot(recall, precision)
        axs[i][j].set_xlabel('Recall')
        axs[i][j].set_ylabel('Precision')
        axs[i][j].set_title(assays[assay_idx])
    
    def plot_recall_precision(self, y_pred_proba:np.array, extra_index:bool = True)->None:
        """
        Given the y scoring for a model, will print out the precision-recall curve.
        """

        fig, axs = plt.subplots(4,3,figsize = (20,20))
        
        for i in range(0,len(self.assays)):
            plot_index_I = int(i/3)
            plot_index_J = i - 3*int(i/3)
            index_tuple = (plot_index_I, plot_index_J)
            if extra_index:
                yy_pred = y_pred_proba[i][:,1]
            else:
                yy_pred = y_pred_proba[:,i]
            self._plot_precision_recall_curve(self.y[:,i],yy_pred, i, index_tuple, axs, self.assays)