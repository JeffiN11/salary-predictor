from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db
from app.models.prediction import PredictionLog
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionLogsResponse,
    ModelInfoResponse,
)
from app.services import model_service
from app.services.ollama_service import get_salary_explanation

router = APIRouter()


@router.post("/", response_model=PredictionResponse)
async def predict_salary(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Predict salary based on job profile.
    Returns predicted salary, range, and an AI explanation.
    """
    try:
        result = model_service.predict(
            job_title=request.job_title,
            location=request.location,
            education=request.education,
            years_experience=request.years_experience,
            skills=request.skills,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    explanation = await get_salary_explanation(
        job_title=request.job_title,
        location=request.location,
        education=request.education,
        years_experience=request.years_experience,
        skills=request.skills,
        predicted_salary=result["predicted_salary"],
    )

    log = PredictionLog(
        job_title=request.job_title,
        location=request.location,
        education=request.education,
        years_experience=request.years_experience,
        num_skills=len(request.skills),
        predicted_salary=result["predicted_salary"],
        explanation=explanation,
        model_version=result["model_version"],
    )
    db.add(log)
    await db.commit()

    return PredictionResponse(
        predicted_salary=result["predicted_salary"],
        salary_range_low=result["salary_range_low"],
        salary_range_high=result["salary_range_high"],
        explanation=explanation,
        model_version=result["model_version"],
    )


@router.get("/logs", response_model=PredictionLogsResponse)
async def get_prediction_logs(db: AsyncSession = Depends(get_db)):
    """Get all past salary predictions."""
    total = (await db.execute(select(func.count(PredictionLog.id)))).scalar_one()
    logs = (await db.execute(
        select(PredictionLog).order_by(PredictionLog.created_at.desc())
    )).scalars().all()
    return PredictionLogsResponse(total=total, predictions=list(logs))


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get current model metadata and evaluation metrics."""
    metadata = model_service.get_metadata()
    if not metadata:
        raise HTTPException(status_code=503, detail="Model not loaded. Run ml/train.py first.")
    return ModelInfoResponse(
        model_name=metadata["model_name"],
        metrics=metadata["metrics"],
        training_samples=metadata["training_samples"],
        feature_cols=metadata["feature_cols"],
    )
