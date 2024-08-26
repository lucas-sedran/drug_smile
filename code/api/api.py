from fastapi import FastAPI, UploadFile, Request,File,HTTPException,Form
from code._01_preprocessing.vect_preproc import vect_preprocess_data
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


# Loading model
model = load()


@app.post("/model")
async def model(request:Request):
    model = await request.json()
    selected_open = model.get('selected_model')
    return {"model" : {selected_open}}


@app.post("/loaddata")
async def predict( model_name: str = Form(...) ,file : UploadFile = File(...)):


    contents = await file.read()
    df = pd.read_parquet(io.BytesIO(contents))
    print(df)

    print(model_name)


    return {"message":"Parquet received","columns": df.columns.tolist()}



@app.post("/predict")
async def predict(file : UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_parquet(io.BytesIO(contents))
    print(df)

    preproc_df = vect_preprocess_data(df)
    prediction = model.predict(preproc_df)

    return {"message":"Parquet received","columns": df.columns.tolist(),"pred":prediction.tolist()}
