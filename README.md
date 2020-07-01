# Drug Discovery - Tox 21

This repository is an attempt to create a model for the Tox-21 data challenge, which resolved in 2014. The data can be obtained from the webpage:

https://tripod.nih.gov/tox21/challenge/data.jsp

In particular, the data used in this repository is the complete dataset found under the subheader 'Training Datasets' in the webpage. The winners of the challenge were announced here:

https://ncats.nih.gov/news/releases/2015/tox21-challenge-2014-winners

In this repository, a model is created to try and predict the signal and stress panels as dictated by the webpage, which is done by means of feature engineering using basic ML practices, but also by making use of known computational chemistry libraries and deep neural networks.


# Introduction

The challenge revolves around the idea of applying machine learning to drug discovery in order to generally shorten the time it takes for drugs to make it to pharmaceutical testing, and then to market. The general idea is that enzymes have so-called active sites, which consume subtrates in which certain desired properties are implemented. The hope is that further chemical reactions may occur in which further required properties occur.

In the context of the dataset here, there are a list of candidate molecules and their performance (active or in-active) on a set of so-called assays. Put more simply - the input dataset is a list of molecules. The output are 12 dimensional arrays, filled with 1's or 0's. The point of the exercise is to be able to predict all 1's and 0's.

The input molecules are represented by so-called SMILES. https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system which is a way to represent a molecule linearly by treating the 2D diagrammatic representation as a graph and performing a depth first search algorithm (roughly). An example would be `CC[C@H](O1)CC[C@@]12CCCO2`. In order to obtain useful information about the molecule, the rdkit library is used extensively https://www.rdkit.org/docs/index.html. The author of this repository makes no claim to understanding the chemistry concepts in full, however, the rdkit library has made it relatively simple to make some progress despite this fact.

Finally, it is also worth noting that for the training portion of this study, the use of google colab was employed - this will be pointed out when certain notebooks are discussed.

# Outline of folder structure
Here, the basic outline of the folder structure is provided:

There are three main folders:
1. `drugdiscovery` : This folder contains all of the preprocesing, DNN training and testing code and encapsulates the main functionailities used in this projet.

2. `data` : This folder contains all of the data that is collected or engineered throughout the project, since this data has come from the Tox-21 group, the data have not been given here.

3. `models `: This folder containts the models that were obtained.

There are 6 notebooks:

1. `data_integrity.ipynb` is the first touch with the downloaded data and provides very rudimentary consistency checks on the data (this file is partnered with the script `clean_data.py` to be described shortly).

2. `data_sourcing.ipynb` provides some details of the kind of data processing and engineering available in regards to the smile data (this is partnered with `preprocess.py`)

3. `EDA.ipynb` performs exploratory data analysis on the smile data as well as all the engineered data.

4. `Benchmark_models.ipynb` provides benchmkark models for the data.

5. `Deep neural net modelling.ipynb` is where the DNN model instanciation adn training occurs, this is done on google colab.

6. `Grid esarch analysis and seeking final model.ipynb` goes through the grid search studied in the previus notebook, identifies an optimal model subject to the area under the precision-recall curve.

There are two python scripts:

1. `clean_data.py` cleans the data subject to the findings in `data_integrity.ipynb`.

2. `preprocess.py` provides data engineering efforts subject to the findings in `data_sourcing.ipynb`.

# The Data

The data itself presents as smile representations, together with molecular weight, various identifiers (from PubChem, Tox-21, etc.) together with the known assay results. After removing the identifiers, the last 5 rows of the table are :

 | SMILES                                |   SR-HSE |   NR-AR |   SR-ARE |   NR-Aromatase |   NR-ER-LBD |   NR-AhR |   SR-MMP |   NR-ER |   NR-PPAR-gamma |   SR-p53 |   SR-ATAD5 |   NR-AR-LBD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CCCc1cc(=O)[nH]c(=S)[nH]1             |        0 |       0 |        0 |              0 |           0 |        0 |        0 |       0 |               0 |        0 |          0 |           0 
| S=C1NCCN1                             |        0 |       0 |        1 |              0 |           0 |        0 |        0 |       0 |               0 |        0 |          0 |           0 |
| S=C1NCCN1                             |        0 |       0 |        0 |              0 |           0 |        0 |        0 |       0 |               0 |        0 |          0 |           0 |
| CCOP(=S)(OCC)Oc1ccc([N+](=O)[O-])cc1  |        0 |       0 |        0 |              0 |           0 |        1 |        0 |       0 |               0 |        0 |          0 |           0 |
| CCC(COC(=O)CCS)(COC(=O)CCS)COC(=O)CCS |        0 |       0 |        0 |              0 |           0 |        0 |   0 |       0 |               0 |        1 |          0 |           0 |

The SMILES are the input data, whilst the remaining columns are the 12 target columns. 

There are two important data integrity issues that need to be addressed:

1. There are duplicate columns, with the same smile but different (or improved, by virtue of NaN->non-NaN or 0->1) targets. The assumption is that further experimental results have increased the table knowledge or modified it. However, the relevant rows have not been modified by the experiment reporter - but a new row have simply been added. This is addressed in `data_intgrity.ipynb` and resolved in `clean_data.py`.

2. Null are a natural consequence of the way the the data was collected. There are two many null values to simply remove them (since they occur intertwined with relevatn and wanted target results), thus they must be worked with. The descision here is bring these data points along the entire process, but to introduce bit masking into the weightings for metric and criterion loss computations in order to not let these values contribute. In the computation of metrics, this is done by enforcing a 2D weighting matrix (which is the bit mask) via, the example line 58 of `drugdiscovery/testing.py`:

        
        precision = precision_score(self.y[:,i],y_pred[:,i], sample_weight = self.sample_weights[:,i])


In the training of the DNN, it is important that that the null values aren't backpropagated - again, this can bed one by making use of the weighting functionality within PyTorch:
       
            # step 5. Re-adjust the criterion weight to mask null values in the validation target set.
            self.criterion.weight = torch.tensor(weight_test).to(self.device).float()
            loss = self.criterion(preds.to(self.device), y_val.to(self.device).float()).to(self.device)
            vloss = loss

which can be found in line 204 of `drugdiscovery/deeplearning.py`.

# Processes

In this section the process of the project is described, this also describes the order that one should read this project. There are 5 clear steps of the project lifecycle:

1. **Data Cleaning and engineering**

    The data is checked for consistency and data engineering opportunities are invesitgated. In this particular case, duplicates were removed and a strategy for dealing with the null values was initiated. For data engineering, the rdkit library was used to generate:
    * Fingerprint data
    * Dice similarities between known toxicophores and fingerprints of scaffold molecules.
    * Molecular descriptors
    
    `data_integrity.ipynb` and `data_sourcing.ipynb` describe cleaning and engineering opporunities within the dataset, whilst `clean_data.py` and `preprocess.py` perform these steps.
    
2. **Exploratory Data Analysis**

    Given the engineered data, a basic exploratory study into the dataset is performed. The structure and balance of the target dataset is studied. The key questions asked here are regarding the distributions of the various input datasets when split by te target dataset.

    A very key takeaway from this study is that the target data in naturally very imbalanced, thus the key metric for success for any models would be the **Area under precision-recall** since, both the precision and the recall focus on how relevant the results are and whether relevatn results are returned at all.

    This is performed in `EDA.ipynb`, at this juncture its also worth looking at `drugdiscovery/testing.py` where a testing framework is built.

3. **Benchmarking**

    Basic benchmarking are also performed, and provides a basic sense of how the testing framework can be utilised. The model parameters are despoted in `/models`. The two models explored are:
    * RandomForeset : The generic model given by sklearn is chosen which has 

           RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

    * XGBClassifier by XGBoost: Again, the generic model given byxgboost:

           XGBClassifier(base_score=None, booster=None, colsample_bylevel=None,
              colsample_bynode=None, colsample_bytree=None, gamma=None,
              gpu_id=None, importance_type='gain', interaction_constraints=None,
              learning_rate=None, max_delta_step=None, max_depth=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=None, num_parallel_tree=None,
              objective='binary:logistic', random_state=None, reg_alpha=None,
              reg_lambda=None, scale_pos_weight=None, subsample=None,
              tree_method=None, validate_parameters=False, verbosity=None)

    This is performed in `Benchmark_models.ipynb`, using a **bag of words** approach, where the results are also displayed. The result of the test set for the random forest are:

Assay| Precision |    Recall |       F1 |    AUPRC |   Accuracy |   Balanced Accuracy |   ROC_AUC 
|---|---|---|---|---|---|---|---|
SR-HSE        |    0.5      | 0.059322  | 0.106061 | 0.175205 |   0.946069 |            0.52797  |  0.52797  
NR-AR         |    0.803922 | 0.418367  | 0.550336 | 0.530519 |   0.972709 |            0.707062 |  0.707062 
SR-ARE        |    0.68254  | 0.133127  | 0.222798 | 0.454856 |   0.846861 |            0.560451 |  0.560451 
NR-Aromatase  |    0.769231 | 0.0934579 | 0.166667 | 0.295378 |   0.948927 |            0.545919 |  0.545919 
NR-ER-LBD     |    0.710526 | 0.243243  | 0.362416 | 0.406449 |   0.959729 |            0.619175 |  0.619175 
NR-AhR        |    0.771084 | 0.243346  | 0.369942 | 0.535456 |   0.901802 |            0.616819 |  0.616819 
SR-MMP        |    0.738318 | 0.265993  | 0.391089 | 0.609741 |   0.873652 |            0.624512 |  0.624512 
NR-ER         |    0.581395 | 0.192308  | 0.289017 | 0.351963 |   0.882521 |            0.586339 |  0.586339 
NR-PPAR-gamma |    0.5      | 0.0333333 | 0.0625   | 0.184745 |   0.97235  |            0.516193 |  0.516193 
SR-p53        |    0.533333 | 0.0567376 | 0.102564 | 0.268889 |   0.938353 |            0.526726 |  0.526726 
SR-ATAD5      |    0.857143 | 0.0674157 | 0.125    | 0.303089 |   0.964706 |            0.53349  |  0.53349  
NR-AR-LBD     |    0.763158 | 0.386667  | 0.513274 | 0.551939 |   0.97593  |            0.691297 |  0.691297 

4. **Grid searching DNN models**

Having established benchmarks, DNN modelling is performed. Particularly, by grid searching over types of data inputs, the layers of the network and whether PCA should be used as a final step to halve the data input dimension. The grid search parameters investigated are:
* Data inputs over all subsets of fingerprint data, dice similarities of toxicophores with the data and molecular descriptors.
* The layers [1024],[1024,2048],[1024,2048,4196]
* Whether a final PCA should be performed just before the data is consumed by the model

The data which is not grid searched, but remains static throughout is:
* Early stopping at a patience level of 10 over the area under precision-recall curve which is computed every 10 epochs.
* Batch size is 128.
* A residual layer in the network itself.
* Adam optimiser with learning rate 4e-5, and weight decay at 1e-5.
* A scheduler which lowers the learning rate on when a plateau on the validation loss is reached.
* Number of epochs fixd to 200, it was found that for this particular grid search - 200 epochs was enough to determine the relative success of the model from the search parameters chosen.

This is all conducted in `Deep neural net modelling.ipynb` with the help of the functionality built in `drugdiscovery/deeplearning`.


5. **Finding an optimal model subject to the grid search**

The process of finding the optimal model from the grid search by virtue of the AUPRC was found in `Grid search analysis and seeing final model.ipynb`.

According to the maximal area under precision-recall curve - the model with data inputs with truncated SVD fingerprint, dice similarities and molecular descriptors, with layers [1024, 2048] and PCA employed to halve the dataset was the optimal. It has an AUPRC of 0.50923616. On the testing set, the metric table is:


Assay| Precision |    Recall |       F1 |    AUPRC |   Accuracy |   Balanced Accuracy |   ROC_AUC 
|---|---|---|---|---|---|---|---|
SR-HSE|0.4|0.277778|0.327869|0.341085|0.937879|0.62687|0.62687
NR-AR|0.666667|0.454545|0.540541|0.526299|0.977212|0.72382|0.72382
SR-ARE|0.670103|0.601852|0.634146|0.641071|0.874372|0.768206|0.768206
NR-Aromatase|0.62963|0.515152|0.566667|0.58764|0.956811|0.748788|0.748788
NR-ER-LBD|0.454545|0.285714|0.350877|0.442262|0.948179|0.634021|0.634021
NR-AhR|0.689655|0.526316|0.597015|0.678326|0.919403|0.748006|0.748006
SR-MMP|0.717391|0.680412|0.698413|0.777485|0.90339|0.813837|0.813837
NR-ER|0.536585|0.293333|0.37931|0.414401|0.884984|0.629425|0.629425
NR-PPAR-gamma|0.272727|0.176471|0.214286|0.205877|0.965891|0.581866|0.581866
SR-p53|0.576923|0.3|0.394737|0.481543|0.932353|0.64127|0.64127
SR-ATAD5|0.466667|0.304348|0.368421|0.428118|0.966667|0.646435|0.646435
NR-AR-LBD|0.5|0.5625|0.529412|0.592103|0.976744|0.774554|0.774554

# Conclusion

The main conclusion to draw is that the discovered DNN model performed a little better that out of the box RandomForest. This appears to be largely due to the not having a large enough feature space to choose from, this is particularly stark given the winning solution had almost 40000 data inputs: https://arxiv.org/pdf/1503.01445.pdf.

For this project to progress - we have to avenues to pursue 
1. Consider polynomial features, or further data engineer processes.
2. Enhance the input dataset, particularly with specific domain knoweledge.
3. Consider a larger variety of nueral network architecture - the most attractive of these would probably be from NLP application since there are structure sequence data. 

# To Do:

1. The challenge has a separate testing dataset and has now released results - thus we should apply our model to this testing dataset.

2. Add polynomial features to the input data.

3. Add RNN and Transformer models.

4. Create a requirements.yaml file.

5. Send trained model to data folder.

6. Need more data.


