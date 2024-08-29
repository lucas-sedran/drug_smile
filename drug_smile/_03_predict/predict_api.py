import os
import pandas as pd
import joblib
from drug_smile._01_preprocessing.vect_preproc import vect_preprocess_data

from drug_smile._02_model_train.GNN_train import GNN_smiles_to_graph
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from torch_geometric.data import Data
import numpy as np
from io import BytesIO
import base64

def from_smile_to_viz(mol):
    img = Draw.MolToImage(mol)
    return img

def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

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
        # y_prob = model.predict_proba(X)[:, 1]
        # threshold = 0.7
        # y_pred_temp = (y_prob >= threshold).astype(int)
        print(f"Prédiction {name_protein} : {y_pred_temp}")
        y_pred_temp = pd.DataFrame(y_pred_temp, columns=[name_protein])
        df_concatenated_temps = pd.concat([df_concatenated_temps, y_pred_temp], axis=1)

    df_concatenated_temps = df_concatenated_temps.drop(columns=['molecule','ecfp'])

    return df_concatenated_temps

def predict_with_gnn(model, X):
    model.eval()
    predictions = []
    with torch.no_grad():
        for graph in X:
            data = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
            out = model(data)
            prob = torch.sigmoid(out).item()
            predictions.append(int(prob>=0.825))
    return np.array(predictions)


def process_model_predictions(df, name_model):
    if name_model in ["Logistic Regression", "Random Forest"]:
        df_encoded = model_vect_predictions(df,name_model)

    elif name_model == "GNN":
        df_encoded = model_GNN_predictions(df,name_model)

    df_encoded['molecule_image'] = df_encoded['molecule_smiles'].apply(lambda x: from_smile_to_viz(Chem.MolFromSmiles(x)))
    df_encoded['molecule_image'] = df_encoded['molecule_image'].apply(pil_to_base64)
    df_encoded = df_encoded[['id', 'molecule_smiles', 'molecule_image','protein_name','binds','BRD4','HSA','sEH']]
    return df_encoded

def model_GNN_predictions(df, name_model):
    # Préproc
    preproc_df = df['molecule_smiles'].apply(GNN_smiles_to_graph)
    X = preproc_df
    print("Préproc done")
    # DataFrame pour stocker les prédictions finales
    df_concatenated_temps = df
    # Boucle sur les protéines
    for name_protein in ['BRD4', 'HSA', 'sEH']:
        # Charger le modèle
        model_name = f"model_{name_model.replace(' ', '_')}_{name_protein}_all"
        parent_dir = os.path.dirname(os.getcwd())
        chemin_fichier = os.path.join(parent_dir, f"drug_smile/models/{model_name}.pkl")
        if os.path.exists(chemin_fichier):
            model = joblib.load(chemin_fichier)
            print(f"----- {model_name} model loaded -----")
            y_pred_temp = predict_with_gnn(model, X)
            print(f"Prédiction {name_protein} : {y_pred_temp}")
            y_pred_temp = pd.DataFrame(y_pred_temp, columns=[name_protein], index=df.index)
            df_concatenated_temps = pd.concat([df_concatenated_temps, y_pred_temp], axis=1)
        else:
            print(f"Le modèle pour {name_protein} n'a pas été trouvé à l'emplacement : {chemin_fichier}")

    return df_concatenated_temps



if __name__ == "__main__":
    # Exemple d'utilisation de la fonction
    parent_dir = os.path.dirname(os.getcwd())
    file_path = os.path.join(parent_dir, f'drug_smile/raw_data/test_5.parquet')
    df = pd.read_parquet(file_path)
    model_name = "Logistic Regression" #Logistic Regression #GNN
    df_result = process_model_predictions(df, model_name)
    # Affichage des résultats
    print(df_result)
