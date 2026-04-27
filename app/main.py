from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.db.database import engine, Base
from app.routers import predictions
from app.services import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    model_service.load_model()
    yield


app = FastAPI(
    title="Salary Predictor API",
    description="MLOps-grade salary prediction API. Predicts salary using XGBoost/Random Forest and explains results using a local LLM (Ollama).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router, prefix="/predict", tags=["Predictions"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Salary Predictor API is running"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
