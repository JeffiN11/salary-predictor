import pytest
from unittest.mock import patch, AsyncMock

MOCK_PREDICTION = {
    "predicted_salary": 105000.0,
    "salary_range_low": 94500.0,
    "salary_range_high": 115500.0,
    "model_version": "XGBoost",
    "num_skills": 3,
}

MOCK_EXPLANATION = "An ML Engineer in Toronto with a Master degree and 3 years experience commands a high salary due to strong demand for AI talent in the Canadian market."

MOCK_METADATA = {
    "model_name": "XGBoost",
    "metrics": {"mae": 4200.0, "rmse": 5800.0, "r2": 0.96},
    "training_samples": 4000,
    "feature_cols": ["job_title_enc", "location_enc", "education_enc", "years_experience", "num_skills"],
}

SAMPLE_REQUEST = {
    "job_title": "ML Engineer",
    "location": "Toronto",
    "education": "Master",
    "years_experience": 3,
    "skills": ["Python", "TensorFlow", "Docker"],
}


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_predict_salary(client):
    with patch("app.routers.predictions.model_service.predict", return_value=MOCK_PREDICTION), \
         patch("app.routers.predictions.get_salary_explanation", new_callable=AsyncMock, return_value=MOCK_EXPLANATION):
        response = await client.post("/predict/", json=SAMPLE_REQUEST)

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_salary"] == 105000.0
    assert data["salary_range_low"] == 94500.0
    assert data["salary_range_high"] == 115500.0
    assert data["explanation"] == MOCK_EXPLANATION
    assert data["model_version"] == "XGBoost"


@pytest.mark.asyncio
async def test_predict_logs_prediction(client):
    with patch("app.routers.predictions.model_service.predict", return_value=MOCK_PREDICTION), \
         patch("app.routers.predictions.get_salary_explanation", new_callable=AsyncMock, return_value=MOCK_EXPLANATION):
        await client.post("/predict/", json=SAMPLE_REQUEST)

    logs_response = await client.get("/predict/logs")
    assert logs_response.status_code == 200
    data = logs_response.json()
    assert data["total"] == 1
    assert data["predictions"][0]["job_title"] == "ML Engineer"
    assert data["predictions"][0]["predicted_salary"] == 105000.0


@pytest.mark.asyncio
async def test_get_prediction_logs_empty(client):
    response = await client.get("/predict/logs")
    assert response.status_code == 200
    assert response.json()["total"] == 0


@pytest.mark.asyncio
async def test_model_info(client):
    with patch("app.routers.predictions.model_service.get_metadata", return_value=MOCK_METADATA):
        response = await client.get("/predict/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "XGBoost"
    assert data["metrics"]["r2"] == 0.96


@pytest.mark.asyncio
async def test_model_info_not_loaded(client):
    with patch("app.routers.predictions.model_service.get_metadata", return_value={}):
        response = await client.get("/predict/model-info")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_predict_model_not_loaded(client):
    with patch("app.routers.predictions.model_service.predict", side_effect=RuntimeError("Model not loaded")):
        response = await client.post("/predict/", json=SAMPLE_REQUEST)
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_generate_data():
    from ml.generate_data import generate_dataset
    df = generate_dataset()
    assert len(df) == 5000
    assert "salary" in df.columns
    assert "job_title" in df.columns
    assert df["salary"].min() >= 40000
