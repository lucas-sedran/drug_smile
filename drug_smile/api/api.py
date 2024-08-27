from fastapi import FastAPI, UploadFile, Request,File,HTTPException,Form
import pandas as pd
from drug_smile._01_preprocessing import *
import io
import joblib
from drug_smile.params import *
from google.cloud import storage

app = FastAPI()

@app.get("/ping")
def home():
    return {"message":"pong"}



@app.get("/")
def root():
    return {'message': 'Hello'}


def load_model(name_file):

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
            print("✅ model downloaded from cloud storage")
        except:
            print(f"\n❌ No {name_file} found in GCS bucket {BUCKET_PROD_NAME} ❌")
            os.remove(model_path)

            return None
    else :
        print("Model already in local folder")
        pass
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f" ### Unable to load model {type}: {e} ###")
    return model



@app.on_event("startup")
async def startup_event():
    print("*"*50)
    print(" Startup API =) ")
    print("*"*50)

    name_file = "model_vect_Logistic_Regression_BRD4_all.pkl"
    app.state.model_vect_Logistic_Regression_BRD4_all = load_model(name_file)

    name_file = "model_GNN_BRD4_all.pkl"
    app.state.model_GNN_BRD4_all = load_model(name_file)



@app.post("/predict")
async def predict( model_name: str = Form(...) ,file : UploadFile = File(...)):

    contents = await file.read()
    df = pd.read_parquet(io.BytesIO(contents))
    print(df)

    print(model_name)

    if model_name == "Logistic Regression":
        model = app.state.model_vect_Logistic_Regression_BRD4_all
        preproc_df = vect_preprocess_data(df)
        X = preproc_df['ecfp'].tolist()
        prediction = model.predict(X)
        return prediction

    if model_name == "Random Forest":
        return { "message" : "From API : Random Forest"}


    if model_name == "GNN":
        model = app.state.model_GNN_BRD4_all
        return { "message" : "From API : GNN Loaded !!!!!"}


    return {"message":"Parquet received","columns": df.columns.tolist(),"model_selected":model_name}
