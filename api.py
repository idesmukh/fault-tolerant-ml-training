from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
from model import SolarPowerPredictionLSTM
from checkpoint import load_checkpoint
import os

model = None

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)