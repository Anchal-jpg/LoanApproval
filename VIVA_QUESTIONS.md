# Loan Approval MLOps Viva Questions

## Machine learning

### 1. What is the objective of this project?

To predict whether a loan application will be approved or rejected from applicant financial and personal information.

### 2. Which dataset is used?

The project uses the Kaggle Loan Approval Dataset by `rohitgrewal`. The target column is `loan_status`, containing `Approved` and `Rejected` values.

### 3. What is the target encoding?

`Approved` is encoded as `1` and `Rejected` as `0`.

### 4. Why is `loan_id` removed?

It is an identifier, not a meaningful applicant feature. Keeping it could introduce noise and accidental patterns.

### 5. Which algorithm is used?

Class-balanced Logistic Regression is used because this is a binary classification problem and the model is fast, interpretable, and suitable for a production baseline.

### 6. Why is preprocessing inside a Pipeline?

The same transformations must be applied during both training and prediction. A scikit-learn Pipeline prevents training-serving preprocessing mismatch.

### 7. Why is One-Hot Encoding used?

Columns such as education and self-employment are categorical. One-Hot Encoding converts their values into numerical features without assuming an artificial order.

### 8. Why is StandardScaler used?

Numeric columns have very different ranges. Scaling prevents large-value features from dominating Logistic Regression.

### 9. Which metrics are tracked?

Accuracy, precision, recall, F1-score, and ROC-AUC are tracked. Precision and recall are especially important because loan decisions have financial consequences.

### 10. What is the purpose of `class_weight="balanced"`?

It gives more importance to the smaller class during training and reduces bias toward the majority class.

## MLOps and MLflow

### 11. Why is MLflow used?

MLflow tracks experiments, parameters, evaluation metrics, artifacts, and registered model versions. This makes model training reproducible and auditable.

### 12. What is the MLflow experiment name?

The experiment is named `loan-approval`.

### 13. What is stored in the MLflow backend?

Run metadata, parameters, metrics, and model registry information are stored in SQLite. Model artifacts are stored in the configured artifact directory.

### 14. What is Model Registry?

Model Registry provides a central place to register, version, and manage trained models.

### 15. What is the MLflow server?

The MLflow server provides a web UI and tracking API. In this project it runs on port `5000`.

### 16. Why must the MLflow image version match the database version?

MLflow database migrations are version-dependent. Using an incompatible server version can cause Alembic errors such as `Can't locate revision identified by ...`.

## API and UI

### 17. What is Streamlit used for?

Streamlit provides an interactive form where a user enters applicant details and receives an approval decision and confidence score.

### 18. What is FastAPI used for?

FastAPI exposes the model through `/predict` and provides `/health` for service monitoring.

### 19. What validation is done by the API?

Pydantic validates ranges such as CIBIL score, loan term, income, and asset values before prediction.

### 20. What does the prediction response contain?

It returns the decision, a Boolean approval value, and the model confidence.

## Docker and CI/CD

### 21. Why is Docker used?

Docker packages the application and its dependencies into a reproducible environment that can run consistently across machines.

### 22. What does Docker Compose provide?

Compose runs the MLflow server, Streamlit UI, and FastAPI service together with consistent networking and port configuration.

### 23. What does GitHub Actions do?

It automatically installs dependencies, trains the model, runs tests, and builds/publishes the Docker image when code is pushed to `main`.

### 24. Why should tests run before Docker publishing?

The build-and-push job depends on the test job. This prevents a broken application from being published as a production image.

### 25. How can this project be improved?

Possible improvements include model approval gates, data validation, drift monitoring, a production database, authentication, explainability, bias checks, and deployment to a cloud platform.
