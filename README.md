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

# Processes

# Model and Results


# To Do:

1. The challenge has a separate testing dataset and has now released results - thus we should apply our model to this testing dataset.

2. Add polynomial features to the input data.

3. Add RNN and Transformer models.

4. Create a requirements.yaml file.

5. Send trained model to data folder.


