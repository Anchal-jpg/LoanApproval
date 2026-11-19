from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "loan_pipeline.joblib"


class LoanApplication(BaseModel):
    no_of_dependents: int = Field(ge=0, le=20)
    education: str
    self_employed: str
    income_annum: float = Field(gt=0)
    loan_amount: float = Field(gt=0)
    loan_term: int = Field(gt=0, le=50)
    cibil_score: int = Field(ge=300, le=900)
    residential_assets_value: float = Field(ge=0)
    commercial_assets_value: float = Field(ge=0)
    luxury_assets_value: float = Field(ge=0)
    bank_asset_value: float = Field(ge=0)


def create_app(model_loader=joblib.load):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not MODEL_PATH.exists():
            raise RuntimeError("Model not found. Run: python src/train.py")
        app.state.model = model_loader(MODEL_PATH)
        yield

    app = FastAPI(title="Loan Approval API", version="1.0.0", lifespan=lifespan)

    @app.get("/")
    def home():
        return {
            "service": "Loan Approval API",
            "status": "ready",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/predict")
    def predict(application: LoanApplication):
        model = getattr(app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        frame = pd.DataFrame([application.model_dump()])
        approved = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0][approved])
        return {
            "decision": "Approved" if approved == 1 else "Rejected",
            "approved": bool(approved),
            "confidence": round(probability, 4),
        }

    return app


app = create_app()
