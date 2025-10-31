from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List
import torch
from model import SolarPowerPredictionLSTM
from checkpoint import load_checkpoint
import os

model = None

class ForecastingInput(BaseModel):
    data: List[List[float]] = Field(..., description="Shape: (24, 8)")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] * 24
            }
        }

class ForecastingOutput(BaseModel):
    forecast: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    checkpoint_dir = './demo_checkpoints'
    best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    
    if os.path.exists(best_path):
        checkpoint_type = 'best'
        print(f"Loading best checkpoint from {best_path}")
    elif os.path.exists(latest_path):
        checkpoint_type = 'latest'
        print(f"Loading latest checkpoint from {latest_path}")
    else:
        print("No checkpoint was found.")
        yield
        return
    
    model = SolarPowerPredictionLSTM()
    model.eval()
    load_checkpoint(checkpoint_type, model, optimizer=None, checkpoint_dir=checkpoint_dir)
    print(f"Model loaded successfully")
    
    yield

app = FastAPI(title="Solar power forecasting API", lifespan=lifespan)

@app.get("/status")
def status_check():
    model_loaded = model is not None
    return {
        "status": "stable",
        "model_loaded": model_loaded
    }

@app.post("/forecast", response_model=ForecastingOutput)
def forecast(input_data: ForecastingInput):
    input_tensor = torch.tensor([input_data.data], dtype=torch.float32)

    with torch.no_grad():
        forecast = model(input_tensor)
    
    return ForecastingOutput(forecast=forecast.item())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)