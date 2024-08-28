# Import Libraries
import pandas as pd
import numpy as np

import os
import rdkit

from rdkit import Chem

from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sklearn import set_config; set_config(display='diagram')

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from drug_smile.params import *
from google.cloud import storage

from drug_smile._01_preprocessing.vect_preproc import vect_load_data, vect_clean_data



## Creation of function to agregate each descriptor of each molecule
def calculate_descriptors(molecule_rdkit):
    descriptors = {}
    for descriptor_name, function in Descriptors.descList:
        descriptors[descriptor_name] = function(molecule_rdkit)
    return descriptors


def cara_check_and_process_file():
    name_file = f"df_cara_preproc_{NAME_PROTEIN}_{NB_SAMPLE}.pkl"
    source_blob_name = f"echantillons/{name_file}"
    # Initialiser le client Google Cloud Storage
    storage_client = storage.Client(project=GCP_PROJECT)
    bucket = storage_client.bucket(BUCKET_DATA_NAME)
    blob = bucket.blob(source_blob_name)

    parent_dir = os.path.dirname(os.getcwd())
    destination_file_name = os.path.join(parent_dir, f'drug_smile/raw_data/{name_file}')

    # Vérifier si le fichier existe dans le bucket
    if blob.exists():
        if os.path.exists(destination_file_name):
            print(f"Le fichier {name_file} existe déjà en local. Transformation en DataFrame en cours...")
            # Charger le fichier en DataFrame
            df_processed = pd.read_pickle(destination_file_name)
        else:
            print(f"Le fichier {name_file} existe déjà dans le bucket. Téléchargement en cours...")
            # Télécharger le fichier du bucket
            blob.download_to_filename(name_file)
            # Charger le fichier en DataFrame
            df_processed = pd.read_pickle(name_file)

        print(f"Le fichier {name_file} a été chargé en DataFrame.")
    else:
        print(f"Le fichier {name_file} n'existe pas dans le bucket. Sauvegarde en cours...")

        df = vect_load_data(NAME_PROTEIN,NB_SAMPLE)
        df = vect_clean_data(df)
        df_chunks = []
        for start in range(0, len(df), CHUNK_SIZE):
            chunk = df.iloc[start:start + CHUNK_SIZE]
            chunk_updated = process_chunk(chunk)
            df_chunks.append(chunk_updated)
        # Concatenate all chunks
        df_processed = pd.concat(df_chunks, ignore_index=True)

        # Sauvegarder df_processed en fichier .pkl localement
        df_processed.to_pickle(destination_file_name)
        print(f"Le fichier {name_file} a été sauvegardé localement.")

        # Uploader le fichier .pkl sur le bucket
        blob.upload_from_filename(destination_file_name)
        print(f"Le fichier {name_file} a été sauvegardé dans le bucket.")

    return df_processed


def process_chunk(chunk):
    # Conversion from SMILE to RDKit Molecule
    chunk['molecule_rdkit'] = chunk['molecule_smiles'].apply(Chem.MolFromSmiles)

    # Features Creation based on RDKit library and Molecule Specificities
    descriptors_df = chunk['molecule_rdkit'].apply(calculate_descriptors).apply(pd.Series)

    # Combine new columns with the chunk
    chunk_updated = pd.concat([chunk, descriptors_df], axis=1)

    return chunk_updated


def cara_preprocess_data():

    df_processed = cara_check_and_process_file()

    # Drop 'binds' and 'molecule_rdkit' columns
    X = df_processed.drop(['binds', 'molecule_rdkit'], axis=1)
    y = df_processed['binds']

    # replace inf values by nan
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # medians computation and default value if median is not available
    default_value = 0
    medians = X.select_dtypes(include='float').median()

    # replace medians not available by default value
    medians.fillna(default_value, inplace=True)

    # replace nan values by medians
    X.fillna(medians, inplace=True)

    # Train Test Split Data
    test_size = 0.3
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    # Define the numerical pipeline
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('minmax_scaler', MinMaxScaler())
    ])

    # Define the categorical pipeline
    cat_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine both pipelines into a preprocessor
    preprocessor = ColumnTransformer([
        ('num_transformer', num_transformer, make_column_selector(dtype_include=[float, int])),
        ('cat_transformer', cat_transformer, make_column_selector(dtype_include=object))
    ])

    # X Train transformed with preprocessor
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    return X_train_transformed, X_val_transformed, y_train, y_val
