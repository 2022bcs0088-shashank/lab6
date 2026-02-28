from fastapi import FastAPI
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

    data.columns = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]

    prediction = model.predict(data)

    return {
        "name": "Shashank Upadhyay", 
        "roll_no": "2022BCS0088", 
        "wine_quality": float(prediction[0])
    }