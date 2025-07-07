# Model: Message Passing Neural Network
# code sourced from: https://keras.io/examples/graph/mpnn-molecular-graphs/

import os
from rdkit import Chem
import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer #older but better with simpler models compared to MolGraphConvFeaturizer
from deepchem.models import MPNNModel
from graph_dataset import GraphListDataset
import numpy as np
import pandas as pd
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #had some errors with JIT -> remove if necessary

drug_df = pd.read_csv('data/results.csv')
smiles_list = drug_df['SMILES'].dropna().astype(str).tolist()

start = time.time()

# Converts SMILES -> RDKit Mol objects
mols = [Chem.MolFromSmiles(smi) for smi in smiles_list] 
print("------MOLS--------")
print(mols)
print("--------------------")

# Remove any None values (invalid SMILES)
valid_pairs = [(smi, mol) for smi, mol in zip(smiles_list, mols) if mol is not None]
valid_smiles = [smi for smi, _ in valid_pairs]
valid_mols = [mol for _, mol in valid_pairs]

# Featurize into graph objects
featurizer = MolGraphConvFeaturizer()

features = []
valid_smiles_filtered = []

for smi, mol in zip(valid_smiles, valid_mols):
    try:
        #featurize each molecule to graph based features
        feat = featurizer.featurize([mol])[0]
        if feat is not None:
            features.append(feat)
            valid_smiles_filtered.append(smi)
    except Exception as e:
        print(f"❌ Failed to featurize: {smi} -> {e}")

# Dummy labels created
dummy_labels = np.zeros(len(features))

# Data structure created for wrapping molecular graphs to integrate to MPNN
dataset = GraphListDataset(X=features, y=dummy_labels, ids=valid_smiles_filtered)

# MPNN model
model = MPNNModel(
    n_tasks=1,
    mode='regression',   # Can be 'classification' too
    number_atom_features = features[0].node_features.shape[1],
    use_queue=False,
    batch_size=32,
)

end = time.time()

print(model.model.summary()) #print the results
print(f"Execution time: {end - start:.4f} seconds")


# message_passing - output shape -> (None, 100) (look at summary output for more details)
# Execution time: 8.4483 seconds
