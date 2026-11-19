import pandas as pd
from fastapi.testclient import TestClient

from src.api import create_app
from src.train import build_pipeline, load_training_data


class FakeModel:
    def predict(self, frame):
        return [1]

    def predict_proba(self, frame):
        return [[0.2, 0.8]]


def test_training_data_loads():
    frame = load_training_data()
    assert len(frame) > 1000
    assert set(frame["loan_status"].unique()) == {0, 1}


def test_pipeline_has_preprocessing_and_classifier():
    frame = load_training_data()
    pipeline = build_pipeline(frame)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_api_prediction():
    app = create_app(model_loader=lambda path: FakeModel())
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert home.json()["service"] == "Loan Approval API"

        response = client.post(
            "/predict",
            json={
                "no_of_dependents": 2,
                "education": "Graduate",
                "self_employed": "No",
                "income_annum": 5000000,
                "loan_amount": 15000000,
                "loan_term": 10,
                "cibil_score": 700,
                "residential_assets_value": 5000000,
                "commercial_assets_value": 3000000,
                "luxury_assets_value": 7000000,
                "bank_asset_value": 4000000,
            },
        )
        assert response.status_code == 200
        assert response.json()["decision"] == "Approved"
