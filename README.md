# salary-predictor 💰

![CI](https://github.com/JeffiN11/salary-predictor/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)

A full MLOps salary prediction pipeline. Trains XGBoost and Random Forest models on realistic salary data, serves predictions via a FastAPI REST API, and uses a local LLM (Ollama + llama3) to explain why a salary is predicted at that level.

## Features

- 🧠 **Model Training** — XGBoost vs Random Forest with cross-validation
- 📊 **Model Evaluation** — MAE, RMSE, R² metrics saved as metadata
- 🚀 **FastAPI serving** — `POST /predict/` returns salary + range
- 🤖 **AI Explanation** — Ollama explains why the salary is predicted
- 📈 **Prediction Logging** — All predictions stored in PostgreSQL
- 📦 **Model Info endpoint** — View current model metrics live
- 🐳 **Docker** — One command to train + serve everything
- ✅ **CI** — GitHub Actions lint and tests

## How It Works

1. `ml/generate_data.py` — generates 5000 realistic salary records
2. `ml/train.py` — trains XGBoost + Random Forest, saves best model
3. FastAPI loads the model on startup
4. `POST /predict/` → ML model predicts → Ollama explains → logged to DB

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI 0.115 (async) |
| ML Models | XGBoost + Random Forest (scikit-learn) |
| AI Explanation | Ollama + llama3 (local) |
| Database | PostgreSQL 16 + SQLAlchemy 2 |
| Containerization | Docker + Docker Compose |
| Testing | pytest + pytest-asyncio |
| CI | GitHub Actions |

## Quick Start

```bash
git clone https://github.com/JeffiN11/salary-predictor.git
cd salary-predictor
docker compose up --build
```

API: http://localhost:8000
Docs: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| POST | /predict/ | Predict salary |
| GET | /predict/logs | View prediction history |
| GET | /predict/model-info | View model metrics |

## Example

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "job_title": "ML Engineer",
    "location": "Toronto",
    "education": "Master",
    "years_experience": 3,
    "skills": ["Python", "TensorFlow", "Docker"]
  }'
```

Response:
```json
{
  "predicted_salary": 105000.0,
  "salary_range_low": 94500.0,
  "salary_range_high": 115500.0,
  "explanation": "An ML Engineer in Toronto with a Master degree and 3 years of experience commands a strong salary due to high demand for AI talent and Toronto being a major tech hub in Canada.",
  "model_version": "XGBoost",
  "currency": "CAD"
}
```

## Running Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -v
```

## Training the Model Manually

```bash
python ml/generate_data.py
python ml/train.py
```
