get_files:
	python -c 'from code._00_preparation.preparation import creation_full_data; creation_full_data()'

get_and_save_little_samples:
	python -c "from code._00_preparation.preparation import get_and_save_little_samples; get_and_save_little_samples()"

get_vecteurs_model:
	python -c "from code._02_model_train.registry import main_vecteurs; main_vecteurs(name_protein='${NAME_PROTEIN}',nb_sample='${NB_SAMPLE}')"

get_GNN_model:
	python -c "from code._02_model_train.registry import main_GNN; main_GNN(name_protein='${NAME_PROTEIN}',nb_sample='${NB_SAMPLE}')"


#################### API #########################
run_api:
	uvicorn code.api.api:app --reload

run_api_8080:
	uvicorn code.api.api:app --host 127.0.0.1 --port 8080 --reload
