from code._01_preprocessing.vect_preproc import vect_load_data, vect_clean_data, vect_preprocess_data
from code._02_model_train.vect_train import vect_split_data, vect_train_and_evaluate, vect_save_model, save_param_model
from code._02_model_train.GNN_train import GNN_transform_data, GNN_create_classes
from code._02_model_train.GNN_train import GNN_find_best_model,GNN_save_model
from code._02_model_train.GNN_train import GNN_train,GNN_find_best_params
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


def main_GNN(name_protein,nb_sample):
    print(f"----- get_vecteurs_model {name_protein} {nb_sample} : START -----")

    # Récupération des données
    df = vect_load_data(name_protein,nb_sample)

    # Transformation et division des données
    X_train,X_test,y_train,y_test,train_loader,test_loader,num_edge_features,num_node_features=GNN_transform_data(df)

    # Instanciation des classes
    GNN,GNNModel= GNN_create_classes()

    #Grid Search
    best_params = GNN_find_best_params(GNNModel,X_train, X_test, y_train, y_test,num_node_features,num_edge_features) # GridSearch
    best_model=GNN_find_best_model(GNN,best_params,num_node_features,num_edge_features)

    #Entrainement
    best_model_trained , avg_val_precision = GNN_train(best_model,best_params,train_loader,test_loader)

    #Saving
    GNN_save_model(best_model_trained,name_protein,nb_sample)
    save_param_model('GNN', avg_val_precision, best_params)

    print(f"----- get_vecteurs_model {name_protein} {nb_sample} : STOP -----\n")

def main_GNN_just_train(best_params, name_protein, nb_sample):
    print(f"----- get_vecteurs_model {name_protein} {nb_sample} : START -----")

    # Récupération des données
    df = vect_load_data(name_protein, nb_sample)

    # Transformation et division des données
    X_train, X_test, y_train, y_test, train_loader, test_loader, num_edge_features, num_node_features = GNN_transform_data(df)

    # Instanciation des classes
    GNN, _ = GNN_create_classes()

    # Utilisation des meilleurs paramètres déjà obtenus
    best_model = GNN_find_best_model(GNN, best_params, num_node_features, num_edge_features)

    # Entraînement
    best_model_trained, avg_val_precision = GNN_train(best_model, best_params, train_loader, test_loader)

    # Sauvegarde
    GNN_save_model(best_model_trained, name_protein, nb_sample)
    save_param_model('GNN', avg_val_precision, best_params)

    print(f"----- get_vecteurs_model {name_protein} {nb_sample} : STOP -----\n")



if __name__ == "__main__":
    nb_sample = NB_SAMPLE
    name_protein = NAME_PROTEIN
    main_vecteurs(name_protein=NAME_PROTEIN, nb_sample=NB_SAMPLE)
