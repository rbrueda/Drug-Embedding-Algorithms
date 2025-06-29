import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import AllChem

# Load SMILES strings from CSV
drug_smiles_df = pd.read_csv('data/results.csv')
smiles_list = drug_smiles_df['SMILES'].dropna().astype(str).tolist()

def morgan_fingerprint_embed(smiles_list):
    mol_list = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    # Compute Mol objects to bit vectors (fingerprints)
    embeddings = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in mol_list]
    return embeddings

start = time.time()
embeddings = morgan_fingerprint_embed(smiles_list)
end = time.time()

print("Embeddings shape:", embeddings.shape)
print(f"Execution time: {end - start:.4f} seconds")

#! Mol objects failed to convert
