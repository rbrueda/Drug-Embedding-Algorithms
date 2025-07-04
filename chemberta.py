import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import time

# Use GPU for speed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(device)
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM", trust_remote_code=True)

# Load SMILES strings from CSV
drug_smiles_df = pd.read_csv('data/results.csv')
smiles_list = drug_smiles_df['SMILES'].dropna().astype(str).tolist()

def chemberta_embed(smiles_list, batch_size=8, max_length=256):
    all_embeddings = []
    # Iterate through batches to avoid too much memory being used
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Extract the result as the last layer
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    # Stacks all lists together vertically
    return np.vstack(all_embeddings)

start = time.time()
embeddings = chemberta_embed(smiles_list)
end = time.time()

print("Embeddings shape:", embeddings.shape)
print(f"Execution time: {end - start:.4f} seconds")

# Output:
# Embeddings shape: (1338, 384)
# Execution time: 0.9239 seconds
