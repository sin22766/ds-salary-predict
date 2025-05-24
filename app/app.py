from fastapi import FastAPI

from app.models.predict import PredictInput, PredictOutput

app = FastAPI(title="DA Salary Predict")


@app.get("/", tags=["Health"])
def read_root():
    return {"message": "Welcome to the FastAPI app"}


@app.get("/predict")
def predict(input: PredictInput) -> PredictOutput:
    result = {"result": input.data[0] + 100}
    return result
