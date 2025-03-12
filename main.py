from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Load the trained model...
with open('./model/model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

# Add CORS middleware to allow requests from any origin...
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FishInput(BaseModel):
    Weight: float
    Length1: float
    Length2: float
    Length3: float
    Height: float
    Width: float
@app.get("/predict")
async def get_predict_info():
    return {"message": "Please send a POST request with fish feature values to get a prediction."}

# Serve the frontend (index.html)
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_species(input_data: FishInput):
    
    input_dict = input_data.model_dump()
    
    features_df = pd.DataFrame([input_dict])
    
    # Make a prediction...
    predicted_species = model.predict(features_df)
    
    return {"predicted_species": predicted_species[0]}

'''if __name__ == "__main__":
    uvicorn.run("main:app", host = "0.0.0.0", port = 8000)'''
