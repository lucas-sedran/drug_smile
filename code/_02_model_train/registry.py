from code._01_preprocessing.vect_preproc import vect_load_data, vect_clean_data, vect_preprocess_data
from code._02_model_train.vect_train import vect_split_data, vect_train_and_evaluate, vect_save_model
from code.params import *

def main_vecteurs(name_protein,nb_sample):
    print(f"----- get_vecteurs_model {name_protein} {nb_sample} : START -----")
    # Récupération et prétraitement des données
    df = vect_load_data(name_protein,nb_sample)
    df = vect_clean_data(df)
    df = vect_preprocess_data(df)

    # Division des données
    X_train, X_val, y_train, y_val = vect_split_data(df)

    # Entraînement et évaluation
    best_model, name_model = vect_train_and_evaluate(X_train, X_val, y_train, y_val)

    # Sauvegarde du meilleur modèle
    vect_save_model(name_model, best_model, name_protein, nb_sample)
    print(f"----- get_vecteurs_model {name_protein} {nb_sample} : STOP -----\n")

if __name__ == "__main__":
    nb_sample = NB_SAMPLE
    name_protein = NAME_PROTEIN
    main_vecteurs(name_protein=NAME_PROTEIN, nb_sample=NB_SAMPLE)
