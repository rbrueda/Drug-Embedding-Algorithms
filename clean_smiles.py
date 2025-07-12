import pandas as pd

df = pd.read_csv("data/results.csv")

clean_df = pd.DataFrame(columns=['DrugBank_ID','PubChem_CID','SMILES'])

for index, row in df.iterrows():
    if not pd.isna(row["SMILES"]):
        clean_df = clean_df._append(row)


clean_df.to_csv("cleaned_drugbank_smiles_mapping.csv", index=False)