#################### 00 Preparation des données #########################
get_files:
# Création du .parquet équilibré :
# Il contient toutes les combinaisons molécules-protéines avec liaison (target = 1) + le même nombre de combinaison mais sans liaison (target = 0).
	python -c 'from drug_smile._00_preparation.preparation import creation_full_data; creation_full_data()'

get_and_save_little_samples:
# Création de petits samples de données : 1k, 5k, 10k et 100k par protéine
	python -c "from drug_smile._00_preparation.preparation import get_and_save_little_samples; get_and_save_little_samples()"

#################### 01 Preprocessing des données #########################
get_vecteurs_preproc:
# Récupération d'un sample de données pour une protéine (stockage local ou dans un bucket de GCP)
# Préprocessing en fonction d'une type d'algo qu'on va utiliser
	python -c "from drug_smile._01_preprocessing.vect_preproc import vect_check_and_process_file; vect_check_and_process_file()"

#################### 02 Entrainement des modèles #########################
get_vecteurs_Grid_Search:
# Réalise un Grid Search sur 3 algo différents : Regression Logistique, Random Forest et SVC
# Sauvegarde le meilleur modèle
	python -c "from drug_smile._02_model_train.registry import main_vect_Grid_search; main_vect_Grid_search(name_protein='${NAME_PROTEIN}',nb_sample='${NB_SAMPLE}')"

get_vecteurs_model:
# Entraine et evalue un modèle en particulier parmi : Regression Logistique ou Random Forest
	python -c "from drug_smile._02_model_train.registry import main_vecteurs; main_vecteurs(name_protein='${NAME_PROTEIN}',nb_sample='${NB_SAMPLE}')"

get_GNN_model:
# Entraine et evalue un modèle GNN
	python -c "from drug_smile._02_model_train.registry import main_GNN; main_GNN(name_protein='${NAME_PROTEIN}',nb_sample='${NB_SAMPLE}')"

get_GNN_model_just_train:
# Entraine un modèle GNN
	python -c "from drug_smile._02_model_train.registry import main_GNN_just_train; main_GNN_just_train(best_params={'hidden_channels': 128,'learning_rate': 0.001,'num_layers': 2}, name_protein='$$NAME_PROTEIN', nb_sample='$$NB_SAMPLE')"

get_cara_model:
# Entraine un modèle SVC
	python -c "from drug_smile._02_model_train.registry import main_cara; main_cara(name_protein='${NAME_PROTEIN}',nb_sample='${NB_SAMPLE}')"


#################### 03 Prédiction #########################
get_vect_predictions:
# Lance la prédiction avec le modèle Logistic Regression
	python -c "from drug_smile._03_predict.predict_api"

#################### API #########################
run_api_8010:
	uvicorn drug_smile.api.api:app --host 127.0.0.1 --port 8010 --reload
