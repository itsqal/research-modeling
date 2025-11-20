from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fuzzy_engine import RiskInferenceEngine

app = FastAPI(title="Fisherman Risk Assessment API", version="1.0")
engine = RiskInferenceEngine()

class FishermanInput(BaseModel):
    weighted_sum_norm: float = Field(..., ge=0, le=1, description="Pattern Match Score (0-1)")
    province: int = Field(..., description="Province Name (e.g., 'Aceh', 'D.I Yogyakarta')")
    how_long_used_signs_years: float = Field(..., ge=0, description="Years experience")
    fisherman_age: float = Field(..., ge=15, description="Age of fisherman")
    how_often_used_signs: float = Field(..., ge=0, description="Frequency index")

    class Config:
        schema_extra = {
            "example": {
                "weighted_sum_norm": 0.15,
                "province": "Aceh",
                "how_long_used_signs_years": 5.0,
                "fisherman_age": 24.0,
                "how_often_used_signs": 10.0
            }
        }

@app.get("/")
def home():
    return {"message": "Fuzzy Risk Engine is Online."}

@app.post("/predict")
def predict_risk(input_data: FishermanInput):
    """
    Calculates risk based on Fuzzy Decision Tree Logic.
    """
    # Convert Pydantic model to dict
    data_dict = input_data.model_dump()
    
    result = engine.predict(data_dict)
    
    if result['status'] == 'error':
        raise HTTPException(status_code=500, detail=result)
        
    return result