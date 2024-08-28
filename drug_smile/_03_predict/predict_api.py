import os
import pandas as pd
import joblib
from drug_smile._01_preprocessing.vect_preproc import vect_preprocess_data
from rdkit import Chem
from rdkit.Chem import Draw
def from_smile_to_viz(mol):
    img = Draw.MolToImage(mol)
    return img
def model_vect_predictions(df,name_model):
    # Préproc
    preproc_df = vect_preprocess_data(df)
    print("Préproc done")
    # Predict
    X = preproc_df['ecfp'].tolist()  # Convertir la colonne 'ecfp' en liste de listes
    df_concatenated_temps = preproc_df
    # Boucle sur les protéines
    for name_protein in ['BRD4', 'HSA', 'sEH']:
        # Charger le modèle
        model_name = f"model_vect_{name_model.replace(' ','_')}_{name_protein}_all"
        parent_dir = os.path.dirname(os.getcwd())
        chemin_fichier = os.path.join(parent_dir, f"drug_smile/models/{model_name}.pkl")
        model = joblib.load(chemin_fichier)
        print(f"----- {model_name} model loaded -----")
        # Prédictions
        y_pred_temp = model.predict(X)
        print(f"Prédiction {name_protein} : {y_pred_temp}")
        y_pred_temp = pd.DataFrame(y_pred_temp, columns=[name_protein])
        df_concatenated_temps = pd.concat([df_concatenated_temps, y_pred_temp], axis=1)
    df_concatenated_temps = df_concatenated_temps.drop(columns=['molecule','ecfp'])
    return df_concatenated_temps

def process_model_predictions(df, name_model):
    """
    Cette fonction traite les prédictions du modèle en fonction du fichier de données et du nom du modèle.
    Arguments :
    - file : Chemin vers le fichier .parquet contenant les données.
    - name_model : Nom du modèle à utiliser (par exemple, "Logistic Regression", "Random Forest", "GNN").
    Retour :
    - Un DataFrame avec les résultats concaténés et les colonnes encodées en one-hot.
    """
    # # Ajout des images
    # df['molecule_image'] = df['molecule_smiles'].apply(lambda x: from_smile_to_viz(Chem.MolFromSmiles(x)))
    # df = df[['id', 'molecule_smiles', 'molecule_image', 'protein_name', 'binds']]
    if name_model in ["Logistic Regression", "Random Forest"]:
        df_encoded = model_vect_predictions(df,name_model)
        return df_encoded
    elif name_model == "GNN":
        # Logique pour le modèle GNN
        for name_protein in ['BRD4', 'HSA', 'sEH']:
            model_file = f"model_{name_model.replace(' ','_')}_{name_protein}_all"
if __name__ == "__main__":
    # Exemple d'utilisation de la fonction
    parent_dir = os.path.dirname(os.getcwd())
    file_path = os.path.join(parent_dir, f'drug_smile/raw_data/test_5.parquet')
    df = pd.read_parquet(file_path)
    model_name = "Logistic Regression"
    df_result = process_model_predictions(df, model_name)
    # Affichage des résultats
    print(df_result)
