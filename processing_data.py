# source: https://github.com/Nowrin19/ddi-hypergraph-project

import pandas as pd
import requests
import sys
import time
import json

file_path = r'data/input/ChCh-Miner_durgbank-chem-chem.tsv.gz'
df = pd.read_csv(file_path, sep='\t', header=None, names=['drug1', 'drug2'])

print("✅ Total interactions loaded:", len(df))

# Extract Unique DrugBank IDs
unique_ids = pd.unique(df[['drug1', 'drug2']].values.ravel())
print("✅ Total unique DrugBank IDs:", len(unique_ids))

# Load Error Files
cid_error_file = open("cid_errors.txt", "w")

def get_cid_from_drugbank(drugbank_id):
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drugbank_id}/cids/JSON'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['IdentifierList']['CID'][0]
    except Exception as e:
        print(f"❌ Error with CID for {drugbank_id}: {e}")
    return None

def get_smiles_from_cid(cid):
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except Exception as e:
        print(f"⚠️ Error with SMILES for CID {cid}: {e}")
    return None

def get_smiles_list(results):
    return [item['SMILES'] for item in results if item['SMILES']]

# Process All DrugBank IDs → Get CID and SMILES
results = []

for i, drug_id in enumerate(unique_ids):
    print(f"🔄 Processing {i+1}/{len(unique_ids)}: {drug_id}")
    sys.stdout.flush()

    cid = get_cid_from_drugbank(drug_id)
    if cid:
        smiles = get_smiles_from_cid(cid)
        if smiles is None:
            print(f"⚠️  CID found but SMILES missing for {drug_id}")
    else:
        print(f"❌ No CID found for {drug_id}")
        #Print to External File for Tracking
        cid_error_file.write(f"❌ No CID found for {drug_id}\n")
        smiles = None

    results.append({
        'DrugBank_ID': drug_id,
        'PubChem_CID': cid,
        'SMILES': smiles
    })

    # Rest Between Each HTTP Request
    time.sleep(0.3)
print(results)
# Dump Results Data to CSV
df_results = pd.DataFrame(results)
df_results.to_csv('data/results.csv', index=False)




