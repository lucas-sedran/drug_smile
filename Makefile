get_files:
	python -c 'from code._00_preparation.preparation import creation_full_data; creation_full_data()'

get_and_save_little_samples:
	python -c "from code._00_preparation.preparation import get_and_save_little_samples; get_and_save_little_samples()"




#################### API #########################
run_api:
	uvicorn code.api.api:app --reload
