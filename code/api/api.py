from fastapi import FastAPI, UploadFile, Request,File
#from tensorflow.keras.models import load_model
import pandas as pd
import io


app = FastAPI()
#app.state.model = load_model()


@app.get("/ping")
def home():
    return {"ping":"pong"}



@app.get("/")
def root():
    return {'greeting': 'Hello'}

""" def load():
    model_path = ""
    model = load_model(model_path, compile=False)
    return model

# Loading model
model = load() """

def preprocess_parquet(parquet_data):
    data = pd.read_parquet(io.BytesIO(parquet_data))
    return data



@app.post("/model")
async def model(request:Request):
    model = await request.json()
    selected_open = model.get('selected_model')
    return {"model" : {selected_open}}



@app.post("/predict")
async def predict(file : UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_parquet(io.BytesIO(contents))
    print(df)

    return {"message":"Parquet received","columns": df.columns.tolist()}
