from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Abrindo o arquivo do modelo no caminho correto
with open("model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

class Instance(BaseModel):
    data: list

@app.post("/predict")
async def predict(inst: Instance):
    input_arr = np.array([inst.data])
    pred = model.predict(input_arr)
    return {"prediction": pred.tolist()}
