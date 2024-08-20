import duckdb
import pandas as pd
import os


def data_combined(nb_total_sample=1000):
    parent_dir = os.path.dirname(os.getcwd())
    parent_dir = os.path.join(parent_dir, 'drug_smile')
    train_path = os.path.join(parent_dir, 'raw_data/train.parquet')
    test_path = os.path.join(parent_dir, 'raw_data/test.parquet')

    # Nombre de sample limité
    con = duckdb.connect()
    df = con.query(f"""(SELECT *
                            FROM parquet_scan('{train_path}')
                            LIMIT {nb_total_sample})
                            """).df()
    con.close()

    # Création d'une colonne pour chaque protéine
    df_pivot = df.pivot_table(index='molecule_smiles', columns='protein_name', values='binds', fill_value=0).reset_index()

    # Convertir les colonnes en type int
    df_pivot = df_pivot.astype({'BRD4': 'int8', 'HSA': 'int8', 'sEH': 'int8'})

    # Supprimer les colonnes inutiles et conserver une seule ligne par molécule
    df_combined = df.drop(columns=['protein_name', 'binds']).drop_duplicates(subset='molecule_smiles').merge(df_pivot, on='molecule_smiles')

    # Affecter l'index comme nouvelle colonne 'id'
    df_combined['id'] = df_combined.index

    # Enregistrer les DataFrames en fichiers .parquet
    name_train = f'train_{(nb_total_sample/1000):.0f}k.parquet'
    df.to_parquet(os.path.join(parent_dir, f'raw_data/{name_train}'))
    name_train_combined = f'train_combined_{(nb_total_sample/1000):.0f}k.parquet'
    df_combined.to_parquet(os.path.join(parent_dir, f'raw_data/{name_train_combined}'))

    # Obtenir la taille des fichiers
    size_df = os.path.getsize(os.path.join(parent_dir, f'raw_data/{name_train}')) / (1024 * 1024)
    size_df_combined = os.path.getsize(os.path.join(parent_dir, f'raw_data/{name_train_combined}')) / (1024 * 1024)

    # Afficher la taille des fichiers
    print(f'----- Process {(nb_total_sample/1000):.0f}k  -----')
    print(f"Taille de {name_train}: {size_df:.2f} MB")
    print(f"Taille de {name_train_combined}: {size_df_combined:.2f} MB")
    print(f"Réduction: {((size_df-size_df_combined)/size_df*100):.2f} %\n")


data_combined(nb_total_sample=1000)
data_combined(nb_total_sample=5000)
data_combined(nb_total_sample=10000)
