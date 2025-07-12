# Code sourced from: https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct

import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import time

class Molformer:
    def __init__(self):
        #Use GPU for speed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model and tokenizer
        self.model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

    def molformer_xl_embed(self, smiles_list, batch_size=8, max_length=256):
        all_embeddings = []
        # Iterate through batches to avoid too much memory being used
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
        # Stacks all lists together vertically
        return np.vstack(all_embeddings)

#testing in class
molformer = Molformer()
start = time.time()
# Load SMILES strings from CSV
drug_smiles_df = pd.read_csv('data/input/cleaned_drugbank_smiles_mapping.csv')
smiles_list = drug_smiles_df['SMILES'].dropna().astype(str).tolist()
embeddings = molformer.molformer_xl_embed(smiles_list)
end = time.time()

print("Embeddings shape:", embeddings.shape)
print(f"Execution time: {end - start:.4f} seconds")

# Output:
# Embeddings shape: (1338, 768)
# Execution time: 7.2763 seconds
