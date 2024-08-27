from rdkit import Chem
from torch_geometric.data import Data
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
from sklearn.preprocessing import StandardScaler

###def GNN_load_data(name_protein, nb_sample):
###""" Charge les données depuis un fichier parquet basé sur le nombre d'échantillons spécifié. """
    ###parent_dir = os.path.dirname(os.getcwd())
   ### train_path = os.path.join(parent_dir, f'drug_smile/raw_data/df_{name_protein}_{nb_sample}.parquet')
  ###  df = pd.read_parquet(train_path)
  ###  return df""


def GNN_combine_corr_features(atom_features):
    # Sélection des colonnes fortement corrélées
    corr_features = atom_features[:, -4:]  # Dernières 4 colonnes : Hybridization, Chirality, In_Ring, Aromaticity

    # Appliquer la PCA pour réduire la redondance
    pca = PCA(n_components=2)  # Garder 2 composantes principales
    reduced_features = pca.fit_transform(corr_features)

    # Combiner avec les autres features
    final_features = np.hstack([atom_features[:, :-4], reduced_features])
    return final_features

def GNN_smiles_to_features(smiles):
    scaler = StandardScaler()
    mol = Chem.MolFromSmiles(smiles)
    atom_features = []

    for atom in mol.GetAtoms():
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        num_hydrogens = atom.GetTotalNumHs()
        atomic_number = atom.GetAtomicNum()

        chirality = atom.GetChiralTag()
        hybridization = atom.GetHybridization()
        in_ring = atom.IsInRing()
        aromaticity = atom.GetIsAromatic()

        atom_features.append([
            degree,
            formal_charge,
            num_hydrogens,
            atomic_number,
            int(chirality),
            int(hybridization),
            int(in_ring),
            int(aromaticity)
        ])

    atom_features = np.array(atom_features)
    scaled_features = scaler.fit_transform(atom_features[:, :5])  # Normaliser uniquement les 5 premières colonnes

    atom_features_final = np.hstack([scaled_features, atom_features[:, 5:]])  # Combiner les colonnes normalisées et non normalisées

    # Combiner les features corrélées
    atom_features_final = GNN_combine_corr_features(atom_features_final)


    return atom_features_final

def GNN_smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_features_final = GNN_smiles_to_features(smiles)

    x = torch.tensor(atom_features_final, dtype=torch.float)

    # Obtenir les paires de noeuds connectés par des liaisons (les arêtes)
    edge_index = []
    edge_features = []
    for bond in mol.GetBonds(): ## GetBonds donne la liste des liaisons
        i = bond.GetBeginAtomIdx() # Donne l'index du début de la liaison
        j = bond.GetEndAtomIdx() # Pareil pour la fin
        edge_index.append([i, j]) # A lire arête de i vers j
        edge_index.append([j, i])  # Réciproque pcq non orienté, nécessaire

        bond_type = bond.GetBondType()
        bond_is_conjugated = bond.GetIsConjugated()  # Conjugaison
        bond_is_in_ring = bond.IsInRing()
        stereochemistry = bond.GetStereo()

        bond_type_dic = {str(Chem.rdchem.BondType.SINGLE):1,
                         str(Chem.rdchem.BondType.DOUBLE):2,
                         str(Chem.rdchem.BondType.TRIPLE):3,
                         str(Chem.rdchem.BondType.AROMATIC):4}
        bond_type=bond_type_dic[str(bond_type)]

        edge_features.append([
            bond_type,  # Type de liaison
            bond_is_conjugated,  # Conjugaison
            int(stereochemistry),
            bond_is_in_ring # Dans un cycle ou non
            ])

        edge_features.append([
        bond_type,  # Type de liaison
        bond_is_conjugated,
        int(stereochemistry),# Conjugaison
        bond_is_in_ring  # Dans un cycle ou non
    ])  ## Deux fois parce que dim 2


    # Convertir en tenseur + transposer pour compatibilité Pytorch geo / contiguous : pour l'efficacité du calcul
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)


    return Data(x=x, edge_index=edge_index,edge_attr=edge_attr) # Data : format du graph en torch geo
