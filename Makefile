get_files:
	python -c 'from code._00_preparation.preparation import creation_full_data; creation_full_data()'

get_and_save_little_samples:
	python -c "from code._00_preparation.preparation import get_and_save_little_samples; get_and_save_little_samples()"

get_vecteurs_Grid_Search:
	python -c "from code._02_model_train.registry import main_vect_Grid_search; main_vect_Grid_search(name_protein='${NAME_PROTEIN}',nb_sample='${NB_SAMPLE}')"

get_vecteurs_preproc:
	python -c "from code._01_preprocessing.vect_preproc import vect_check_and_process_file; vect_check_and_process_file()"

get_vecteurs_model:
	python -c "from code._02_model_train.registry import main_vecteurs; main_vecteurs(name_protein='${NAME_PROTEIN}',nb_sample='${NB_SAMPLE}')"

get_GNN_model:
	python -c "from code._02_model_train.registry import main_GNN; main_GNN(name_protein='${NAME_PROTEIN}',nb_sample='${NB_SAMPLE}')"


get_GNN_model_just_train:
	python -c "from code._02_model_train.registry import main_GNN_just_train; main_GNN_just_train(best_params={'hidden_channels': 128,'learning_rate': 0.001,'num_layers': 2}, name_protein='$$NAME_PROTEIN', nb_sample='$$NB_SAMPLE')"

get_cara_model:
	python -c "from code._02_model_train.registry import main_cara; main_cara(name_protein='${NAME_PROTEIN}',nb_sample='${NB_SAMPLE}')"

#################### API #########################
run_api_8010:
	uvicorn code.api.api:app --host 127.0.0.1 --port 8010 --reload

