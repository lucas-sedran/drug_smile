import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def vect_load_data(name_protein, nb_sample):
    """ Charge les données depuis un fichier parquet basé sur le nombre d'échantillons spécifié. """
    parent_dir = os.path.dirname(os.getcwd())
    train_path = os.path.join(parent_dir, f'drug_smile/raw_data/df_{name_protein}_{nb_sample}.parquet')
    df = pd.read_parquet(train_path)
    return df

def vect_clean_data(df):
    """ Supprime les doublons et les colonnes inutiles du DataFrame. """
    print(f"Nombre de duplicat : {int(df.duplicated().sum()/len(df))}")  # Nb de duplicats
    df.drop_duplicates(inplace=True)  # Supprimer les duplicats
    total_nan = df.isna().sum().sum()
    print(f"Nombre total de NaN dans le DataFrame : {total_nan}")
    df = df.drop(columns=['buildingblock1_smiles', 'buildingblock2_smiles' , 'buildingblock3_smiles'])
    return df

def vect_generate_ecfp(molecule, radius=2, bits=1024):
    """ Génère un vecteur de bits représentant la molécule en fonction de la structure chimique locale
    autour de chaque atome jusqu'à une certaine distance (radius). """
    if molecule is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits)
    return list(fingerprint)

def vect_preprocess_data(df):
    """ Applique le prétraitement sur le DataFrame : convertit les SMILES en objets RDKit et génère les ECFP. """
    df['molecule'] = df['molecule_smiles'].apply(Chem.MolFromSmiles)
    df['ecfp'] = df['molecule'].apply(vect_generate_ecfp)
    return df
