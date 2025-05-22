from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="DA Salary Predict")


class Input(BaseModel):
    data: List[float]

class Output(BaseModel):
    result: float


@app.get("/", tags=["Health"])
def read_root():
    return {"message": "Welcome to the FastAPI app"}

@app.get("/predict",response_model=Output)
def predict(input: Input):
    result = {"result": input.data[0]+100}
    # result = {"result": 0.123123}
    return result
