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


def load_data(train_path, protein_name, binds):
    con = duckdb.connect()
    df = con.query(f"""
        SELECT *
        FROM parquet_scan('{train_path}')
        WHERE binds = {binds} AND protein_name = '{protein_name}'
    """).df()
    con.close()
    return df

def count_molecules(df, molecule_column='molecule_smiles'):
    unique_count = df[molecule_column].nunique()
    return unique_count

def define_path():
    parent_dir = os.path.dirname(os.getcwd())
    train_path = os.path.join(parent_dir, 'drug_smile/raw_data/train.parquet')
    return parent_dir, train_path

def creation_data_bind():
    parent_dir, train_path = define_path()

    # Chargement des données
    df_BRD4 = load_data(train_path, 'BRD4', 1)
    df_HSA = load_data(train_path, 'HSA', 1)
    df_sEH = load_data(train_path, 'sEH', 1)

    # Concaténation des DataFrames
    df_bound = pd.concat([df_BRD4, df_HSA, df_sEH])
    df_bound = df_bound.sort_values(by='id').reset_index(drop=True)

    # Affichage des informations
    print(f'Nombre de liaisons molécule/protéine : {df_bound.shape[0]}')
    print(f"Nombre de molécules uniques : {count_molecules(df_bound)}")

    # Pivot des données
    df_bound_pivot = df_bound.pivot_table(index='molecule_smiles', columns='protein_name', values='binds', fill_value=0)
    df_bound_pivot = df_bound_pivot.astype({'BRD4': 'int8', 'HSA': 'int8', 'sEH': 'int8'})
    df_bound_combined = df_bound.drop(columns=['protein_name', 'binds']).drop_duplicates(subset='molecule_smiles').merge(df_bound_pivot, on='molecule_smiles')

    # Comptage des liaisons
    df_bound_combined['nb_binds'] = df_bound_combined[['BRD4', 'HSA', 'sEH']].sum(axis=1)

    for n in range(4):
        count = df_bound_combined[df_bound_combined['nb_binds'] == n].shape[0]
        print(f"Nb de molécules avec {n} liaison(s) : {count} ({round(count/df_bound_combined['molecule_smiles'].nunique()*100,2)}%)")

    # Filtrage des données avec 1 seule liaison
    df_bound_combined_filtered = df_bound_combined[df_bound_combined['nb_binds'] == 1].drop('nb_binds', axis=1)

    # Ajout de la colonne 'protein_name'
    df_bound_combined_filtered['protein_name'] = df_bound_combined_filtered.apply(lambda row: row[['BRD4', 'HSA', 'sEH']].idxmax() if row[['BRD4', 'HSA', 'sEH']].sum() == 1 else None, axis=1)
    df_bound_combined_filtered['binds'] = 1
    df_bound_combined_filtered = df_bound_combined_filtered.drop(['BRD4', 'HSA', 'sEH'], axis=1)
    return df_bound_combined_filtered

def completed_data(df_bound):
    # Création des DataFrames pour les protéines avec 0 liaison
    parent_dir, train_path = define_path()

    con = duckdb.connect()
    for protein in ['BRD4', 'HSA', 'sEH']:
        df_positive = df_bound[df_bound['protein_name'] == protein]
        df_negative = con.query(f"""
            SELECT *
            FROM parquet_scan('{train_path}')
            WHERE binds = 0 AND protein_name = '{protein}'
            ORDER BY random()
            LIMIT {df_positive.shape[0]}
        """).df()
        df_final = pd.concat([df_positive, df_negative], axis=0)
        df_final['binds'] = df_final['binds'].astype('int8')

        # Save as .parquet
        df_final.to_parquet(os.path.join(parent_dir, f'drug_smile/raw_data/df_{protein}_all.parquet'))

        print(f"Nb de {protein}_0 : {df_negative.shape[0]}")
    con.close()


def creation_full_data():
    print('----- creation_full_data : START -----')
    df_bound = creation_data_bind()
    completed_data(df_bound)
    print('----- creation_full_data : DONE -----\n')




def get_little_samples(nb_sample):
    parent_dir = os.path.dirname(os.getcwd())
    BRD4_path = os.path.join(parent_dir, 'drug_smile/raw_data/df_BRD4_all.parquet')
    HSA_path = os.path.join(parent_dir, 'drug_smile/raw_data/df_HSA_all.parquet')
    sEH_path = os.path.join(parent_dir, 'drug_smile/raw_data/df_sEH_all.parquet')

    con = duckdb.connect()
    my_list = []
    for my_path in [BRD4_path, HSA_path,sEH_path]:
        df = con.query(f"""(SELECT *
                                FROM parquet_scan('{my_path}')
                                WHERE binds = 0
                                ORDER BY random()
                                LIMIT {round(nb_sample/2,0)})
                                UNION ALL
                                (SELECT *
                                FROM parquet_scan('{my_path}')
                                WHERE binds = 1
                                ORDER BY random()
                                LIMIT {round(nb_sample/2,0)})""").df()
        df = df.drop('__index_level_0__', axis=1)
        my_list.append(df)
    con.close()
    return my_list

def save_little_samples(my_list):
    [df_BRD4, df_HSA, df_sEH] = my_list
    nb = df_BRD4.shape[0]
    parent_dir = os.path.dirname(os.getcwd())
    df_BRD4.to_parquet(os.path.join(parent_dir, f'drug_smile/raw_data/df_BRD4_{int(nb/1000)}k.parquet'))
    df_HSA.to_parquet(os.path.join(parent_dir, f'drug_smile/raw_data/df_HSA_{int(nb/1000)}k.parquet'))
    df_sEH.to_parquet(os.path.join(parent_dir, f'drug_smile/raw_data/df_sEH_{int(nb/1000)}k.parquet'))

def get_and_save_little_samples():
    print('----- get_and_save_little_samples : START -----')
    save_little_samples(get_little_samples(nb_sample=1000))
    save_little_samples(get_little_samples(nb_sample=5000))
    save_little_samples(get_little_samples(nb_sample=10000))
    save_little_samples(get_little_samples(nb_sample=100000))
    print('----- get_and_save_little_samples : DONE -----\n')



if __name__ == "__main__":
    creation_full_data()

# data_combined(nb_total_sample=1000)
# data_combined(nb_total_sample=5000)
# data_combined(nb_total_sample=10000)
