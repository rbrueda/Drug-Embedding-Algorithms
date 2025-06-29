# Model: Message Passing Neural Network
# to use with SMILES, convert SMILES to molecular graphs

from rdkit import Chem
import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.models import MPNNModel
import numpy as np
import pandas as pd
import torch

drug_df = pd.read_csv('data/results.csv')
smiles_list = drug_df['SMILES'].dropna().astype(str).tolist()

featurizer = MolGraphConvFeaturizer()
# Converts SMILES -> RDKit Mol objects
mols = [Chem.MolFromSmiles(smi) for smi in smiles_list] 

# Remove any None values (invalid SMILES)
valid_pairs = [(smi, mol) for smi, mol in zip(smiles_list, mols) if mol is not None]
valid_smiles = [smi for smi, _ in valid_pairs]
valid_mols = [mol for _, mol in valid_pairs]

# Featurize into graph objects
featurizer = MolGraphConvFeaturizer()
features = featurizer.featurize(valid_mols)

# Create dummy labels (required by DeepChem Dataset)
dummy_labels = np.zeros(len(features))

# Create DeepChem Dataset
dataset = dc.data.NumpyDataset(X=features, y=dummy_labels)

# Initialize MPNN model
model = MPNNModel(
    n_tasks=1,
    mode='regression',   # Can be 'classification' too
    number_atom_features=features[0].get_atom_features().shape[1],
    use_queue=False,
    batch_size=32
)

# Run a dummy training loop to initialize weights (no real training)
model.fit(dataset, nb_epoch=1)

# Extract embeddings 
embeddings = model.predict_embedding(dataset)

print("MPNN Embeddings shape:", embeddings.shape)


#! issue: no data generated -> zero size array