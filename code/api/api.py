from fastapi import FastAPI, UploadFile
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

def preprocess_csv(csv_data):
    data = pd.read_parquet(io.BytesIO(csv_data))
    return data

@app.post("/predict")
async def predict(file : UploadFile):
    csv = await file.read()

    data = preprocess_csv(csv)
    csv_dict = data.to_dict()
    print(csv_dict)
    #predict = model.predict(data)
    return {"data":csv_dict}
