# NOTE: Tested code using Python v3.10.12 for DeepChem code v2.8.0

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from newmodels import BaseGGNN 
class MPNN:
    def __init__(self):
        pass

    def atom_features(self, atom):
        # Simple atom features example: atomic number one-hot vector (up to Z=100)
        Z = atom.GetAtomicNum()
        features = torch.zeros(100) #start with 100 features per molecule and change accordingly
        features[Z-1] = 1
        return features

    def bond_features(self, bond):
        # Simple bond type one-hot (single, double, triple, aromatic)
        bt = bond.GetBondType()
        features = torch.zeros(4)
        if bt == Chem.rdchem.BondType.SINGLE:
            features[0] = 1
        elif bt == Chem.rdchem.BondType.DOUBLE:
            features[1] = 1
        elif bt == Chem.rdchem.BondType.TRIPLE:
            features[2] = 1
        elif bt == Chem.rdchem.BondType.AROMATIC:
            features[3] = 1
        return features

    def smiles_to_data(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Node features
        atom_feats = [self.atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.stack(atom_feats).float()  # shape: (num_atoms, 100)

        # Edges and edge features
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            print("bond:")
            print(bond)
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])  # add reverse edge for undirected graph?

            bf = self.bond_features(bond)
            edge_attr.append(bf)
            edge_attr.append(bf)

        if len(edge_index) == 0:
            # molecule with no bonds (e.g. single atom) - we need to use these for our implementation in get_weight() method
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(edge_attr).float()

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def mpnn_embed(self, smiles_list):
        data_list = []
        for smi in smiles_list:
            data = self.smiles_to_data(smi)
            if data is not None:
                data_list.append(data)

        loader = DataLoader(data_list, batch_size=32, shuffle=False)

        # Use your BaseGGNN model; set state_size to match your node feature size (100 here)
        model = BaseGGNN(state_size=100, num_layers=3, total_edge_types=4)  # 4 edge types from bond_features

        model.eval()

        all_embeddings = []
        with torch.no_grad():
            for batch in loader:
                # (batch, None, None) since batch contains edge_attr and edge_index features
                embeddings = model.get_weight((batch, None, None), batch_size=batch.num_graphs)
                all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        print("Embeddings shape:", all_embeddings.shape)
        return all_embeddings

