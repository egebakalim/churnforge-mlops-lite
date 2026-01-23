## Churnforge MLOps Lite

End-to-end churn prediction mini-system with:

Great Expectations (data validation gate)

MLflow (experiment tracking + artifacts)

FastAPI (prediction API)

pytest (data + features + API tests)

Setup

```bash
make install
```

Add data

Place the Telco churn dataset at:

data/telco_churn.csv

Target column must be named Churn and contain values like:

Yes / No

or 1 / 0

Initialize + validate data suite

```bash
make ge_init
make ge_validate
```

Train + track experiments

```bash
make train
mlflow ui
```

Then open:
http://localhost:5000

Serve the model

```bash
make serve
```

Endpoints:

GET /health

POST /predict

Example JSON payload:
```
json
{
"tenure": 12,
"MonthlyCharges": 70.35,
"Contract": "Month-to-month",
"InternetService": "Fiber optic"
}
```

Run tests

```bash
make test
make coverage
```

Project structure

```
churnforge-mlops-lite/
src/
data.py
features.py
train.py
serve.py
tests/
test_data.py
test_features.py
test_api.py
data/
telco_churn.csv
great_expectations/
suite.json
validation_result.json
artifacts/
model.pkl
requirements.txt
Makefile
README.md
```
