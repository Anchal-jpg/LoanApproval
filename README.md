# Loan Approval MLOps Project

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![CI](https://github.com/Anchal-jpg/LoanApproval/actions/workflows/ci.yml/badge.svg)

End-to-end loan approval prediction project using a Kaggle dataset, scikit-learn, MLflow, Streamlit, FastAPI, Docker, Docker Compose, and GitHub Actions.

## Project architecture

```text
Kaggle CSV -> preprocessing pipeline -> Logistic Regression
                                      |
                         metrics/model logged to MLflow
                                      |
                 Streamlit UI / FastAPI prediction API
```

The model uses numeric imputation and scaling, categorical imputation and one-hot encoding, followed by a class-balanced Logistic Regression classifier. `loan_id` is excluded because it is only an identifier.

## Dataset

Source: Kaggle - `rohitgrewal/loan-approval-dataset`

The downloaded CSV is stored at `data/raw/loan_approval_dataset.csv`.

## Run locally

```powershell
cd loan-approval-mlops
python -m pip install -r requirements.txt
python src/train.py
streamlit run src/app.py
```

Streamlit: <http://localhost:8501>

API:

```powershell
uvicorn src.api:app --reload --port 8000
```

API docs: <http://localhost:8000/docs>

MLflow UI:

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

MLflow UI: <http://localhost:5000>

## Docker

```powershell
docker compose up --build
```

- Streamlit: <http://localhost:8501>
- API: <http://localhost:8000/docs>
- MLflow: <http://localhost:5000>

To run only MLflow:

```powershell
docker compose up -d mlflow
```

If port `8501` is already occupied, use a different host port:

```powershell
docker run -p 8502:8501 loan-approval-mlops:test
```

Then open <http://localhost:8502>.

## Tests

```powershell
python -m pytest -v
```

## GitHub Actions

The workflow in `.github/workflows/ci.yml`:

1. Installs Python dependencies.
2. Trains the model.
3. Runs automated tests.
4. Builds and publishes a Docker image to GitHub Container Registry on pushes to `main`.

The repository needs package write permission enabled for the workflow to publish to GHCR.

## Repository structure

```text
data/raw/                 Kaggle dataset
src/train.py              Training and MLflow logging
src/app.py                Streamlit UI
src/api.py                FastAPI API
tests/                    Automated tests
models/                   Generated local model (ignored by Git)
.github/workflows/ci.yml  GitHub Actions pipeline
Dockerfile                Streamlit image
docker-compose.yml        MLflow, API and Streamlit services
```

## Viva preparation

See [VIVA_QUESTIONS.md](VIVA_QUESTIONS.md) for common project questions and answers.
