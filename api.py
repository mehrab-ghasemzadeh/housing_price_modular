from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import ast
import traceback
import os
import pandas as pd
from typing import Union
from pred import predict_new

import matplotlib
matplotlib.use('Agg')

BASE_DATA_DIR = "data"
DATA_FILE = "data_final.csv"
OLDER_DATA = "data_final_2y.csv"
NEIGHBORHOODS_LIST = "neighborhood_ids.csv"
BASE_MODEL_DIR = "model_reg"
MODEL_DIR = "best_model.pkl"
SCALER_X_DIR = "scaler_X.pkl"
SCALER_Y_DIR = "scaler_y.pkl"

DATA_2Y = pd.read_csv(os.path.join(BASE_DATA_DIR, OLDER_DATA), index_col=0)
DATA = pd.read_csv(os.path.join(BASE_DATA_DIR, DATA_FILE), index_col=0)
NEIGHBORHOODS = pd.read_csv(os.path.join(BASE_DATA_DIR, NEIGHBORHOODS_LIST), index_col=0)

templates = Jinja2Templates(directory="templates")

app = FastAPI(
    title="تخمین قیمت مسکن",
    description="برآورد قیمت ملک با استفاده از روش های ترکیبی هوش مصنوعی",
    version="0.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

def _load_neighbourhoods(df = NEIGHBORHOODS) -> list[dict]:
    df = df[["id", "title"]]
    return df.to_dict(orient="records")

@app.get("/neighbourhoods")
def get_neighbourhoods():
    """Return list of {id, title} for all neighbourhoods."""
    return _load_neighbourhoods()

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "تخمین قیمت مسکن",
            "neighbourhoods": _load_neighbourhoods(),
        },
    )

class Listing(BaseModel):
    area: float = Field(..., gt=0, description="Total area in square meters")
    num_bedrooms: int = Field(..., ge=0)
    id_neighbourhood: int = Field(..., description="Neighbourhood ID")
    floor: int = Field(..., ge=0)
    parking: int = Field(..., ge=0)
    elevator: int = Field(..., ge=0, le=1)
    storeHouse: int = Field(..., ge=0, le=1)
    construction_year: int = Field(..., ge=0)

class PredictionResult(BaseModel):
    status: int
    content: dict

def _listing_to_query(listing: Listing) -> dict:
    query = listing.model_dump(exclude_none=True)
    query["elevator"] = bool(query["elevator"])
    query["storeHouse"] = bool(query["storeHouse"])
    query["building_age"] = 1405-query['construction_year']
    return query

@app.post("/predict", response_model=PredictionResult)
def predict_endpoint(listings: Listing):
    if not listings:
        raise HTTPException(status_code=422, detail="Request body must contain at least one listing.")

    if not os.path.exists(os.path.join(BASE_MODEL_DIR, MODEL_DIR)):
        raise HTTPException(
            status_code=503,
            detail=f"Ensemble model not found at {os.path.join(BASE_MODEL_DIR, MODEL_DIR)}. Run 'reg.py' first.",
        )
    if not os.path.exists(os.path.join(BASE_MODEL_DIR, SCALER_X_DIR)):
        raise HTTPException(
            status_code=503,
            detail=f"Ensemble model not found at {os.path.join(BASE_MODEL_DIR, SCALER_X_DIR)}. Run 'reg.py' first.",
        )
    if not os.path.exists(os.path.join(BASE_MODEL_DIR, SCALER_Y_DIR)):
        raise HTTPException(
            status_code=503,
            detail=f"Ensemble model not found at {os.path.join(BASE_MODEL_DIR, SCALER_Y_DIR)}. Run 'reg.py' first.",
        )

    try:
        return predict_new(query=_listing_to_query(listings), data_2y=DATA_2Y, data=DATA, neighborhoods=NEIGHBORHOODS_LIST)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )