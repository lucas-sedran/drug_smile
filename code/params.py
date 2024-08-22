import os

##################  VARIABLES  ##################
RANDOM_STATE = 42

BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_TARGET = os.environ.get("MODEL_TARGET")


LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),"code","model")
