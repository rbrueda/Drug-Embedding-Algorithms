# Model: Message Passing Neural Network
# code sourced from: https://keras.io/examples/graph/mpnn-molecular-graphs/
#* ran on Python 3.10.12 for up-to-date deepchem version (2.8.0)

import os
from rdkit import Chem
import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer #older but better with simpler models compared to MolGraphConvFeaturizer
from deepchem.models import MPNNModel
import pandas as pd
from deepchem.data import NumpyDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #had some errors with JIT -> remove if necessary

drug_df = pd.read_csv('data/input/cleaned_drugbank_smiles_mapping.csv')
smiles_list = drug_df['SMILES'].dropna().astype(str).tolist()

# Converts SMILES -> RDKit Mol objects
mols = [Chem.MolFromSmiles(smi) for smi in smiles_list] 

# Featurize into graph objects
featurizer = MolGraphConvFeaturizer()
graph_features = featurizer.featurize(mols)
print(type(graph_features), type(graph_features[0]))

# Convert graph features to a Numpy dataset for compatibility
dataset = NumpyDataset(X=graph_features)
print(type(dataset.X[0]))

# Create our MPNN model
model = MPNNModel(
    n_tasks=1,   
    mode='classification', 
    batch_size=32,
)
# Fit our data to model
model.fit(dataset, nb_epoch=10)
#! error occurs here where MPNN model from deepchem expects different format for GraphData (most likely because it expects legacy code) hence, we will not use it for our paper