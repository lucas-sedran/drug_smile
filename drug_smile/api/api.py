from fastapi import FastAPI, UploadFile, Request,File,HTTPException,Form
import pandas as pd
from drug_smile._03_predict.predict_api import process_model_predictions
import io
import joblib
from drug_smile.params import *
from google.cloud import storage
import os
import pickle


app = FastAPI()


@app.get("/ping")
def home():
    return {"message":"pong"}



@app.get("/")
def root():
    return {'message': 'Hello'}

def load_model(name_file):


    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH


    # Create folders models if not exist
    if not os.path.isdir(MODELS_PATH):
            os.makedirs(MODELS_PATH)
    else:
        pass

    model_path = os.path.join(MODELS_PATH, name_file)
    print(f'{model_path = }')

    if not os.path.exists(os.path.join(MODELS_PATH, name_file)):

        # Initialize Google Cloud Storage
        storage_client = storage.Client(project=GCP_PROJECT)
        bucket = storage_client.bucket(BUCKET_PROD_NAME)
        blob = bucket.blob(name_file)
        try:
            blob.download_to_filename(model_path)
            print(f"✅ model {name_file} downloaded from cloud storage")
        except:
            print(f"\n❌ No {name_file} found in GCS bucket {BUCKET_PROD_NAME} ❌")
            os.remove(model_path)

            return None
    else :
        print(f"Model {name_file} already in local folder")
        pass
    try:
        model = joblib.load(open(model_path,'rb'))
    except Exception:
        raise HTTPException(status_code=500, detail=f" ### Unable to load model {name_file} from {model_path = } ###")
    return model



@app.on_event("startup")
async def startup_event():
    print("*"*50)
    print(" Startup API =) ")
    print("*"*50)

    #Binary
    name_file = "model_vect_Logistic_Regression_BRD4_all.pkl"
    print(f"Loading {name_file} ...")
    app.state.model_vect_Logistic_Regression_BRD4_all = load_model(name_file)
    print(f'✔️✔️✔️{name_file} loaded✔️✔️✔️')

    name_file = "model_vect_Logistic_Regression_HSA_all.pkl"
    print(f"Loading {name_file} ...")
    app.state.model_vect_Logistic_Regression_HSA_all = load_model(name_file)
    print(f'✔️✔️✔️{name_file} loaded✔️✔️✔️')

    name_file = "model_vect_Logistic_Regression_sEH_all.pkl"
    print(f"Loading {name_file} ...")
    app.state.model_vect_Logistic_Regression_sEH_all = load_model(name_file)
    print(f'✔️✔️✔️{name_file} loaded✔️✔️✔️')

    #Graphs
    name_file = "model_GNN_BRD4_all.pkl"
    print(f"Loading {name_file} ...")
    app.state.model_GNN_BRD4_all = load_model(name_file)
    print(f'✔️✔️✔️{name_file} loaded✔️✔️✔️')

    name_file = "model_GNN_HSA_all.pkl"
    print(f"Loading {name_file} ...")
    app.state.model_GNN_HSA_all = load_model(name_file)
    print(f'✔️✔️✔️{name_file} loaded✔️✔️✔️')

    name_file = "model_GNN_sEH_all.pkl"
    print(f"Loading {name_file} ...")
    app.state.model_GNN_sEH_all = load_model(name_file)
    print(f'✔️✔️✔️{name_file} loaded✔️✔️✔️')



@app.post("/predict")
async def predict( model_name: str = Form(...) ,file : UploadFile = File(...)):

    contents = await file.read()
    df = pd.read_parquet(io.BytesIO(contents))
    print(df)
    print(model_name)

    result_df = process_model_predictions(df,model_name)
    print(result_df)
    print(f'{result_df.to_json() =}')
    return result_df.to_json()
