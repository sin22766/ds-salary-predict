from typing import List

from pydantic import BaseModel


class PredictInput(BaseModel):
    data: List[float]

class PredictOutput(BaseModel):
    result: float