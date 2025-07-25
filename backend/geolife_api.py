# backend/geolife_api.py
# Libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import uvicorn
from typing import List, Dict
from geolife_model import load_geolife_model, cluster_geolife_user

app = FastAPI()

cluster_location_key = cluster_geolife_user

model, feature_names = load_geolife_model()

class ClusteringRequest(BaseModel):
    """Input for User-specific Clustering"""
    uid: object
    distance: float
    min_k: int

class NextLocationPredictionRequest(BaseModel):
    """Input for User-specific Next Location Prediction"""
    uid: object
    lat_origin: float
    lng_origin: float
    timedelta: float
    month: int
    day: int
    day_of_week: int
    hour_in_day: int
    minute_in_hour: int

@app.get("/")
def read_root():
    return {"message": "GeoLife Next Location Prediction API", "status": "active"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/cluster")
def cluster_locations(request: ClusteringRequest):

    try:
        print("attempting to cluster")
        df, scores = cluster_geolife_user(request.uid, request.distance, request.min_k)
        api_dict = {
            'df': df.to_dict(orient='records'),
            'scores': scores
        }
        return api_dict
    
    except Exception as e: 
        print("Error encountered")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_next_location(request: NextLocationPredictionRequest):
    
    try:
        print("attempting to predict")
        # Convert input to appropriate format for the model
        features = {
            'uid': request.uid,
            'lat_origin': request.lat_origin,
            'lng_origin': request.lng_origin,
            'timedelta': request.timedelta,
            'month': request.month,
            'day': request.day,
            'day_of_week': request.day_of_week,
            'hour_in_day': request.hour_in_day,
            'minute_in_hour': request.minute_in_hour
        }
        
        input_data = [[
            features['uid'],
            features['lat_origin'],
            features['lng_origin'],
            features['timedelta'],
            features['month'],
            features['day'],
            features['day_of_week'],
            features['hour_in_day'],
            features['minute_in_hour']
        ]]
        
        print("processed features")
        # df = pd.DataFrame([d.model_dump() for d in data])
        # print("did dump")
        
        print("Feature names:", feature_names)
        print("Type of feature_names:", type(feature_names))

        
        input_df = pd.DataFrame(input_data, columns=feature_names)
        print("converted to df")
        
        print(input_df)
        
        prediction = model.predict(input_df)
        
        print(prediction)
        
        return prediction
    
    except Exception as e: 
        print("Error encountered")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("geolife_api:app", host="0.0.0.0", port=8000, reload=True)
    pass