import pandas as pd
from chemberta import Chemberta # Import Chemberta class to extract embeddings
from molformer_xl import Molformer # Import Molformer class to extract embeddings
from morgan_fingerprint import MorganFingerprint # Import MorganFingerprint class to extract embeddings
from mpnn import MPNN

# Load SMILES strings from CSV
drug_smiles_df = pd.read_csv('data/input/cleaned_drugbank_smiles_mapping.csv')
smiles_list = drug_smiles_df['SMILES'].dropna().astype(str).tolist()

# Model: ChemBERTa
chemberta = Chemberta()
chemberta_embeddings = chemberta.chemberta_embed(smiles_list)

embedding_rows = []
# Extract embeddings for chemberta
for index, row in drug_smiles_df.iterrows():
    embedding = chemberta_embeddings[index]
    embedding_row = {"DrugBank_ID": row["DrugBank_ID"]}
    for i, val in enumerate(embedding):
        embedding_row[i] = val
    embedding_rows.append(embedding_row)

chemberta_embedding_df = pd.DataFrame(embedding_rows)
chemberta_embedding_df.to_parquet("data/Chemberta-Embeddings.pq", index=False)

# Model: Molformer-XL
molformer = Molformer()
molformer_embeddings = molformer.molformer_xl_embed(smiles_list)

embedding_rows = []
# Extract embeddings for molformer
for index, row in drug_smiles_df.iterrows():
    embedding = molformer_embeddings[index]
    embedding_row = {"DrugBank_ID": row["DrugBank_ID"]}
    for i, val in enumerate(embedding):
        embedding_row[i] = val
    embedding_rows.append(embedding_row)

molformer_embedding_df = pd.DataFrame(embedding_rows)
molformer_embedding_df.to_csv("data/MolFormer-Embeddings.csv", index=False)
molformer_embedding_df.to_parquet("data/MolFormer-Embeddings.pq", index=False)

# Model: Morgan Fingerprint
morgan_fp = MorganFingerprint()
morgan_fp_embeddings = morgan_fp.morgan_fingerprint_embed(smiles_list)

embedding_rows = []
# Extract embeddings for molformer
for index, row in drug_smiles_df.iterrows():
    embedding = morgan_fp_embeddings[index]
    embedding_row = {"DrugBank_ID": row["DrugBank_ID"]}
    for i, val in enumerate(embedding):
        embedding_row[i] = val
    embedding_rows.append(embedding_row)

morgan_fp_embedding_df = pd.DataFrame(embedding_rows)
morgan_fp_embedding_df.to_csv("data/MorganFingerprint-Embeddings.csv", index=False)
morgan_fp_embedding_df.to_parquet("data/MorganFingerprint-Embeddings.pq", index=False)

mpnn = MPNN()
mpnn_embeddings = mpnn.mpnn_embed(smiles_list)

embedding_rows = []
# Extract embeddings for molformer
for index, row in drug_smiles_df.iterrows():
    embedding = mpnn_embeddings[index]
    embedding_row = {"DrugBank_ID": row["DrugBank_ID"]}
    for i, val in enumerate(embedding):
        embedding_row[i] = val.item()
    embedding_rows.append(embedding_row)

mpnn_embedding_df = pd.DataFrame(embedding_rows)
mpnn_embedding_df.to_csv("data/MPNN-Embeddings.csv", index=False)
# For working around error for saving to parquet
mpnn_embedding_df.columns = mpnn_embedding_df.columns.map(str)
mpnn_embedding_df.to_parquet("data/MPNN-Embeddings.pq", index=False)



