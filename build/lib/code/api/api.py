from fastapi import FastAPI, UploadFile, Request,File,HTTPException,Form
from drug_smile._01_preprocessing.vect_preproc import vect_preprocess_data
import pandas as pd
import io
import joblib


app = FastAPI()
#app.state.model = load_model()


@app.get("/ping")
def home():
    return {"ping":"pong"}



@app.get("/")
def root():
    return {'greeting': 'Hello'}

def load():
    global model
    model_path = "/home/dodohellio/code/DodooHellio/drug_smile/models/model_vect_SVC_BRD4_10k.pkl"
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to load model: {e}")
    return model


# Loading model
model = load()



@app.post("/loaddata")
async def predict( model_name: str = Form(...) ,file : UploadFile = File(...)):


    contents = await file.read()
    df = pd.read_parquet(io.BytesIO(contents))
    print(df)

    print(model_name)
    if model_name == "Logistic Regression":
        preproc_df = vect_preprocess_data(df)
        X = preproc_df['ecfp'].tolist()
        prediction = model.predict(X)
        return prediction


    return {"message":"Parquet received","columns": df.columns.tolist(),"model_selected":model_name}



@app.post("/predict")
async def predict(file : UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_parquet(io.BytesIO(contents))
    print(df)

    preproc_df = vect_preprocess_data(df)
    prediction = model.predict(preproc_df)

    return {"message":"Parquet received","columns": df.columns.tolist(),"pred":prediction.tolist()}
