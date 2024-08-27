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
from sklearn.pipeline import Pipeline, make_pipeline
from code.params import *


## Creation of function to agregate each descriptor of each molecule
def calculate_descriptors(molecule_rdkit):
    descriptors = {}
    for descriptor_name, function in Descriptors.descList:
        descriptors[descriptor_name] = function(molecule_rdkit)
    return descriptors


def process_chunk(chunk):
    # Conversion from SMILE to RDKit Molecule
    chunk['molecule_rdkit'] = chunk['molecule_smiles'].apply(Chem.MolFromSmiles)

    # Features Creation based on RDKit library and Molecule Specificities
    descriptors_df = chunk['molecule_rdkit'].apply(calculate_descriptors).apply(pd.Series)

    # Combine new columns with the chunk
    chunk_updated = pd.concat([chunk, descriptors_df], axis=1)

    return chunk_updated

def cara_preprocess_data(df, chunk_size=CHUNK_SIZE):
    df_chunks = []

    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        chunk_updated = process_chunk(chunk)
        df_chunks.append(chunk_updated)

    # Concatenate all chunks
    chunk_updated = pd.concat(df_chunks, ignore_index=True)

    # Drop 'binds' and 'molecule_rdkit' columns
    X = chunk_updated.drop(['binds', 'molecule_rdkit'], axis=1)
    y = chunk_updated['binds']

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
