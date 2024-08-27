from sklearn.svm import SVC
from sklearn.metrics import average_precision_score, classification_report
from code._02_model_train.vect_train import save_param_model
from code.params import *
import joblib
from google.cloud import storage

def cara_save_model(name_model, model, name_protein, nb_sample):
    """ Enregistre le meilleur modèle trouvé. """
    name_fichier_model = f"model_cara_{name_model}_{name_protein}_{nb_sample}.pkl"
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

def run_model_svc(X_train_transformed,X_val_transformed,y_train,y_val):
    #Instantiate model
    model = SVC(C=81, coef0=0, gamma=0.001, kernel='rbf', probability=True)

    # Train the model
    model.fit (X_train_transformed, y_train)

    # Predict y
    y_pred = model.predict(X_val_transformed)

    # Evaluate Average Precision Score
    average_precision_scorer = average_precision_score(y_val, y_pred)

    print('\n------------------------------------------------------------')
    name_model = 'Support Vector Machine'
    print(f"{name_model}:")
    print(f"  - Best Average Precision: {average_precision_scorer:.4f}")
    print(f"  - Best Parameters: {{'C':81, 'coef0':0, 'gamma':0.001, 'kernel':'rbf'}}")

    name_protein = NAME_PROTEIN
    nb_sample = NB_SAMPLE
    cara_save_model(name_model.replace(" ", "_"), model, name_protein, nb_sample)
    save_param_model(name_model + ' (cara)', average_precision_scorer, {'C':81, 'coef0':0, 'gamma':0.001, 'kernel':'rbf'})

    print(classification_report(y_val, y_pred))
    print('------------------------------------------------------------')
