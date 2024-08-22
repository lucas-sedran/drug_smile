from fastapi import FastAPI


app = FastAPI()
#app.state.model = load_model()


@app.get("/ping")
def home():
    return {"ping":"pong"}


@app.get("/")
def root():
    return {'greeting': 'Hello'}


@app.get("/predict")
def predict(X_val):
    print("here")
    pass
