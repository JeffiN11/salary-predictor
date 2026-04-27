from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    job_title: str = Field(..., example="ML Engineer")
    location: str = Field(..., example="Toronto")
    education: str = Field(..., example="Master")
    years_experience: int = Field(..., ge=0, le=40, example=3)
    skills: list[str] = Field(default=[], example=["Python", "TensorFlow", "Docker"])

    model_config = {"from_attributes": True}


class PredictionResponse(BaseModel):
    predicted_salary: float
    salary_range_low: float
    salary_range_high: float
    explanation: Optional[str]
    model_version: str
    currency: str = "CAD"

    model_config = {"from_attributes": True}


class PredictionLogResponse(BaseModel):
    id: int
    job_title: str
    location: str
    education: str
    years_experience: int
    num_skills: int
    predicted_salary: float
    explanation: Optional[str]
    model_version: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionLogsResponse(BaseModel):
    total: int
    predictions: list[PredictionLogResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    metrics: dict
    training_samples: int
    feature_cols: list[str]
