import os
import json
import logging
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "ml/models")

SKILLS = ["Python", "SQL", "AWS", "Docker", "Kubernetes", "React", "FastAPI", "TensorFlow", "Spark", "Go"]

_model = None
_encoders = None
_feature_cols = None
_metadata = None


def load_model():
    global _model, _encoders, _feature_cols, _metadata
    try:
        _model = joblib.load(f"{MODEL_DIR}/model.pkl")
        _encoders = joblib.load(f"{MODEL_DIR}/encoders.pkl")
        _feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
        with open(f"{MODEL_DIR}/metadata.json") as f:
            _metadata = json.load(f)
        logger.info("Model loaded: %s", _metadata.get("model_name"))
    except FileNotFoundError:
        logger.warning("Model not found. Run ml/train.py first.")


def get_metadata() -> dict:
    return _metadata or {}


def predict(
    job_title: str,
    location: str,
    education: str,
    years_experience: int,
    skills: list[str],
) -> dict:
    if _model is None:
        raise RuntimeError("Model not loaded. Run ml/train.py first.")

    row = {
        "job_title_enc": _encoders["job_title"].transform([job_title])[0],
        "location_enc": _encoders["location"].transform([location])[0],
        "education_enc": _encoders["education"].transform([education])[0],
        "years_experience": years_experience,
        "num_skills": len(skills),
    }
    for skill in SKILLS:
        row[f"skill_{skill.lower()}"] = 1 if skill in skills else 0

    df = pd.DataFrame([row])[_feature_cols]
    predicted = float(_model.predict(df)[0])
    predicted = max(30000, round(predicted, -2))

    return {
        "predicted_salary": predicted,
        "salary_range_low": round(predicted * 0.90, -2),
        "salary_range_high": round(predicted * 1.10, -2),
        "model_version": _metadata.get("model_name", "unknown"),
        "num_skills": len(skills),
    }

