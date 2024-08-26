import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from google.cloud import storage
from code.params import *

def download_blob(gcp_project, bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # Crée un client pour interagir avec GCS
    storage_client = storage.Client(project=gcp_project)
    # Accède au bucket spécifié
    bucket = storage_client.bucket(bucket_name)
    # Accède au fichier (blob) dans le bucket
    blob = bucket.blob(source_blob_name)
    # Télécharge le fichier localement
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def vect_load_data(name_protein, nb_sample):
    """ Charge les données depuis un fichier parquet basé sur le nombre d'échantillons spécifié. """
    name_file = f"df_{name_protein}_{nb_sample}.parquet"
    parent_dir = os.path.dirname(os.getcwd())
    train_path = os.path.join(parent_dir, f'drug_smile/raw_data/{name_file}')

    if os.path.exists(train_path):
        print(f"---------------- Data downloaded on local file ----------------")
        df = pd.read_parquet(train_path)
    else:
        print(f"------------------- Data to download on GCP -------------------")
        gcp_project = GCP_PROJECT
        bucket_name = BUCKET_DATA_NAME
        source_blob_name = f"echantillons/{name_file}"
        destination_file_name = os.path.join(parent_dir, f'drug_smile/raw_data/{name_file}')
        download_blob(gcp_project, bucket_name, source_blob_name, destination_file_name)
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

def vect_preprocess_data(df, chunk_size=CHUNK_SIZE):
    """Applique le prétraitement sur le DataFrame par chunks : convertit les SMILES en objets RDKit et génère les ECFP."""
    print(f"------------------- START smile transformation into molecule -------------------")

    df_chunks = []

    # Découpe du DataFrame en chunks
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        print(f"Processing chunk {i // chunk_size + 1}")

        # Transformation des SMILES en molécules RDKit
        chunk['molecule'] = chunk['molecule_smiles'].apply(Chem.MolFromSmiles)
        print(f"------------------- START molecule transformation into ECFP for chunk {i // chunk_size + 1} -------------------")

        # Génération des ECFP
        chunk['ecfp'] = chunk['molecule'].apply(vect_generate_ecfp)
        print(f"------------------- STOP molecule transformation into ECFP for chunk {i // chunk_size + 1} -------------------")
        df_chunks.append(chunk)

    # Concatenation de tous les chunks
    df_processed = pd.concat(df_chunks, ignore_index=True)
    print(f"------------------- FINISHED processing all chunks -------------------")
    return df_processed

def check_and_process_file():
    name_file = f"df_vect_preproc_{NAME_PROTEIN}_{NB_SAMPLE}.pkl"
    source_blob_name = f"echantillons/{name_file}"
    # Initialiser le client Google Cloud Storage
    storage_client = storage.Client(project=GCP_PROJECT)
    bucket = storage_client.bucket(BUCKET_DATA_NAME)
    blob = bucket.blob(source_blob_name)

    # Vérifier si le fichier existe dans le bucket
    if blob.exists():
        print(f"Le fichier {name_file} existe déjà dans le bucket. Téléchargement en cours...")

        # Télécharger le fichier du bucket
        blob.download_to_filename(name_file)

        # Charger le fichier en DataFrame
        df = pd.read_pickle(name_file)
        print(f"Le fichier {name_file} a été chargé en DataFrame.")
    else:
        print(f"Le fichier {name_file} n'existe pas dans le bucket. Sauvegarde en cours...")

        # On réalise le préprocessing
        df = vect_load_data(NAME_PROTEIN,NB_SAMPLE)
        df = vect_clean_data(df)
        df_processed = vect_preprocess_data(df)

        # Sauvegarder df_processed en fichier .pkl localement
        parent_dir = os.path.dirname(os.getcwd())
        destination_file_name = os.path.join(parent_dir, f'drug_smile/raw_data/{name_file}')
        df_processed.to_pickle(destination_file_name)

        # Uploader le fichier .pkl sur le bucket
        blob.upload_from_filename(destination_file_name)
        print(f"Le fichier {name_file} a été sauvegardé dans le bucket.")

        # Charger le DataFrame pour un éventuel traitement ultérieur
        df = df_processed

    return df
