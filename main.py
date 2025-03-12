from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn
import numpy as np
import pandas as pd

# Load the trained model from the pickle file
with open('./model/model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

# Define the expected input schema using Pydantic
class FishInput(BaseModel):
    Weight: float
    Length1: float
    Length2: float
    Length3: float
    Height: float
    Width: float

@app.post("/predict")
async def predict_species(input_data: FishInput):
    # Convert the input data to a dictionary
    input_dict = input_data.model_dump()
    # Create a DataFrame from the dictionary to include valid feature names
    features_df = pd.DataFrame([input_dict])
    
    # Make a prediction using the loaded model
    predicted_species = model.predict(features_df)
    
    return {"predicted_species": predicted_species[0]}

if __name__ == "__main__":
    uvicorn.run("main:app", host = "0.0.0.0", port = 8000)
