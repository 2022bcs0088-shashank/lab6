from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model.pkl")

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def home():
    return {"message": "Wine Quality Prediction API is running!"}

@app.post("/predict")
def predict(features: WineFeatures):
    data = pd.DataFrame([features.dict()])

    data = data.rename(columns={
        "fixed_acidity": "fixed acidity",
        "volatile_acidity": "volatile acidity",
        "citric_acid": "citric acid",
        "residual_sugar": "residual sugar",
        "free_sulfur_dioxide": "free sulfur dioxide",
        "total_sulfur_dioxide": "total sulfur dioxide"
    })

    try:
        prediction = model.predict(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "name": "Shashank Upadhyay", 
        "roll_no": "2022BCS0088", 
        "wine_quality": float(prediction[0])
    }
