import os

##################  VARIABLES  ##################
RANDOM_STATE = 42
NB_SAMPLE = os.environ.get("NB_SAMPLE")
NAME_PROTEIN = os.environ.get("NAME_PROTEIN")
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BUCKET_DATA_NAME = os.environ.get("BUCKET_DATA_NAME")
BUCKET_PROD_NAME = os.environ.get("BUCKET_PROD_NAME")
