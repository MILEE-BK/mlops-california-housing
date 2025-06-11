from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn

# Load model
model = mlflow.sklearn.load_model("model")  # Path to model folder

# Create app
app = FastAPI()

# Define request schema
class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    input_data = [[
        data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
        data.Population, data.AveOccup, data.Latitude, data.Longitude
    ]]
    prediction = model.predict(input_data)[0]
    return {"predicted_price": prediction}
