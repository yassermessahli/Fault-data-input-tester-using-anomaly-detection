from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI()


class ScoringItem(BaseModel):
    Sex: int
    Marital_status: int
    Age: int
    Education: int
    Income: int
    Occupation: int
    Settlement_size: int


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("normalizer.pkl", "rb") as g:
    normalizer = pickle.load(g)


@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    x = np.array([item.Sex, item.Marital_status, np.log(item.Age) + 1, item.Education,
                 np.log(item.Income) + 1, item.Occupation, item.Settlement_size])
    x = normalizer.transform([x])

    p = model.pdf(x) / 65444665.36036717  # max proba
    return {"proba": p}
