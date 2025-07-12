import pandas as pd
import numpy as np
import time
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

class MorganFingerprint:
    def __init__(self):
        pass

    def morgan_fingerprint_embed(self, smiles_list):
        embeddings = [] 
        for smile in smiles_list:
            mol = Chem.MolFromSmiles(smile)
            # Compute Mol objects to bit vectors (fingerprints)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            # Create numpy array
            arr = np.zeros((1024,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            embeddings.append(arr)
        return np.array(embeddings)

# Load SMILES strings from CSV
drug_smiles_df = pd.read_csv('data/results.csv')
smiles_list = drug_smiles_df['SMILES'].dropna().astype(str).tolist()

start = time.time()
morgan_fp = MorganFingerprint()
embeddings = morgan_fp.morgan_fingerprint_embed(smiles_list)
end = time.time()

print("Embeddings shape:", embeddings.shape) 
print(f"Execution time: {end - start:.4f} seconds")

#Embeddings shape: (1338, 1024)
# Execution time: 0.2216 seconds

