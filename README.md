# Drug Discovery - Tox 21

https://ncats.nih.gov/news/releases/2015/tox21-challenge-2014-winners

## Data
DATA AMALGAMATION PROCESS:
The starting point of the data is what one recieves from the webpage (unfortunately, the webpage https://tripod.nih.gov/tox21/challenge/data.jsp where the data may be retrieved is defunct
but it is an .sdf file of the various identifiers (PubChem, Tox21), the smile, molecular weight ). 

The columns of the data is essense takes the form:

* | ID | SMILES |

There are various ID's related to the PubChem database and the Tox21 dataset.
The smiles are a linear representation of the molecules and are a key input for us.

*  SR-HSE | NR-AR | SR-ARE  | NR-Aromatase  | NR-ER-LBD  | NR-AhR   | SR-MMP |  NR-ER |  NR-PPAR-gamma  | SR-p53 |  SR-ATAD5 NR-AR-LBD  |

The columns above are the targets for this dataset. There are 1's and 0's for any of the twelve potential targets. Our goal will be to construct a model which finds these. Since there are multiple 1's and 0's its possible to think of this as a multiple learning task, as well as binary learning for each target.

