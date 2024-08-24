import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, average_precision_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from code.params import *
from google.cloud import storage
import pickle


def vect_split_data(df):
    """ Divise les données en ensembles d'entraînement et de validation. """
    X = df['ecfp'].tolist()  # Convertir la colonne 'ecfp' en liste de listes
    y = df['binds']  # La colonne cible
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    return X_train, X_val, y_train, y_val

def vect_train_and_evaluate(X_train, X_val, y_train, y_val):
    """ Entraîne plusieurs modèles en utilisant GridSearch et évalue leurs performances. """
    param_grid = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, solver='lbfgs'),
            'params': {'C': [0.1, 1, 10]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
        },
        'Support Vector Machine': {
            'model': SVC(probability=True),
            'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        }
    }

    average_precision_scorer = make_scorer(average_precision_score, response_method='predict_proba')
    best_model = None
    best_score = 0
    best_name_model = None

    for name_model, mp in param_grid.items():
        grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring=average_precision_scorer)
        grid.fit(X_train, y_train)

        # Meilleur modèle
        model = grid.best_estimator_
        y_proba = model.predict_proba(X_val)[:, 1]
        ap_score = average_precision_score(y_val, y_proba)

        print('\n------------------------------------------------------------')
        print(f"{name_model}:")
        print(f"  - Best Average Precision: {ap_score:.4f}")
        print(f"  - Best Parameters: {grid.best_params_}")

        if ap_score > best_score:
            best_score = ap_score
            best_model = model
            best_name_model = name_model.replace(" ", "_")

        y_pred = model.predict(X_val)
        # cm = confusion_matrix(y_val, y_pred)
        # plt.figure(figsize=(6, 4))
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        # plt.title(f"Matrice de Confusion - {name}")
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.show()

        name_protein = NAME_PROTEIN
        nb_sample = NB_SAMPLE
        vect_save_model(name_model.replace(" ", "_"), model, name_protein, nb_sample)
        save_param_model(name_model, ap_score, grid.best_params_)

        print(classification_report(y_val, y_pred))
        print('------------------------------------------------------------')

    return best_model, best_name_model

def vect_save_model(name_model, model, name_protein, nb_sample):
    """ Enregistre le meilleur modèle trouvé. """
    name_fichier_model = f"model_vect_{name_model}_{name_protein}_{nb_sample}.pkl"
    gcp_project = GCP_PROJECT
    bucket_name = BUCKET_PROD_NAME
    source_model_name = name_fichier_model
    parent_dir = os.path.dirname(os.getcwd())
    destination_file_name = os.path.join(parent_dir, f'drug_smile/models/{name_fichier_model}')

    # Télécharge le fichier localement
    joblib.dump(model, destination_file_name)
    print(f"\nModèle enregistré à : {destination_file_name}\n")

    # Crée un client pour interagir avec GCS
    storage_client = storage.Client(project=gcp_project)
    # Accède au bucket spécifié
    bucket = storage_client.bucket(bucket_name)
    # Créer un nouvel objet blob dans le bucket
    blob = bucket.blob(source_model_name)
    # Télécharger le fichier local vers GCS
    blob.upload_from_filename(destination_file_name)
    print(f"\nModèle enregistré sur GCS dans le bucket {bucket_name}\n")


def save_param_model(name_model, ap_score, best_params_):
    # Définir les variables nécessaires
    chemin_fichier_local = "models/ours_models.pkl"
    gcp_project = GCP_PROJECT
    bucket_name = BUCKET_PROD_NAME
    name_fichier_model = "ours_models.pkl"

    # Crée un client pour interagir avec GCS
    storage_client = storage.Client(project=gcp_project)
    # Accède au bucket spécifié
    bucket = storage_client.bucket(bucket_name)
    # Créer un nouvel objet blob pour interagir avec le fichier sur GCS
    blob = bucket.blob(name_fichier_model)

    # Vérifier si le fichier existe dans le bucket GCS
    if blob.exists():
        # Télécharger le fichier depuis le bucket GCS vers le répertoire local temporaire
        blob.download_to_filename(chemin_fichier_local)
        # Charger le dictionnaire existant depuis le fichier téléchargé
        with open(chemin_fichier_local, 'rb') as fichier:
            ours_models = pickle.load(fichier)
    else:
        # Si le fichier n'existe pas, initialiser un dictionnaire vide
        ours_models = {}

    # Nouvelles informations à ajouter
    nouvelles_infos = {
        (f"{name_model} {NB_SAMPLE}"): {"Average Precision": round(float(ap_score), 4), "Parameters": best_params_}
    }

    # Mise à jour du dictionnaire existant avec les nouvelles informations
    ours_models.update(nouvelles_infos)

    # Créer le répertoire 'models' s'il n'existe pas encore
    os.makedirs(os.path.dirname(chemin_fichier_local), exist_ok=True)

    # Sauvegarder le dictionnaire mis à jour localement
    with open(chemin_fichier_local, 'wb') as fichier:
        pickle.dump(ours_models, fichier)

    # Uploader le fichier mis à jour dans le bucket GCS
    blob.upload_from_filename(chemin_fichier_local)
    print(f"\nModèle mis à jour et enregistré sur GCS dans le bucket {bucket_name}\n")
