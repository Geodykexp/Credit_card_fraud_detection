# Credit Card Fraud Detection

A project for detecting fraudulent credit card transactions. It provides:
- A trained model serialized to a single pickle file
- A FastAPI service exposing a /score endpoint for predictions
- Scripts/notebooks for training and quick validation
- Docker support for containerized deployment


## Dataset
The project uses the common anonymized credit card transactions dataset from Kaggle.
- Source (hosted copy): https://www.dropbox.com/scl/fi/hh8c3x23wlmi2ra0cdwqv/creditcard.csv?rlkey=lnttjl0nftukwelj5e1cf1zwn&st=bkf1yeot&dl=0
- Note: The file is large and not included in the repository.


## Machine Learning Model
This project uses a scikit-learn model trained on the dataset above. The following models were ran and compared to find the best performing model. They include:
1) Logistic Regression model (lr)
2) Decision Tree model (dtc)
3) RandomForestClassifier model (rf)

The best performing model was the RandomForestClassifier model (rf).s trained on the dataset and serialized to credit_card_fraud_detection.pkl in the following order:
1) DictVectorizer (dv)
2) RandomForestClassifier model (rf)

Key points:
- Features: V1..V28 and Amount (floats)
- Preprocessing: DictVectorizer maps JSON feature dicts to model-ready vectors
- Inference: The API loads dv and rf, transforms incoming JSON, and returns a 0/1 class label

Example (training-side) serialization pattern:
```python
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

# X_dicts: list[dict] of feature mappings, y: labels
# dv = DictVectorizer(sparse=False)
# X = dv.fit_transform(X_dicts)
# rf = RandomForestClassifier(...).fit(X, y)

with open('credit_card_fraud_detection.pkl', 'wb') as f:
    pickle.dump(dv, f)
    pickle.dump(rf, f)
```


## FastAPI Service
The API is implemented in main.py and loads the serialized assets on startup. It exposes:
- GET / — basic health/info
- POST /score — returns {"prediction": 0|1}

Model loading and scoring (as used in main.py):
```python
with open('credit_card_fraud_detection.pkl', 'rb') as f:
    dv = pickle.load(f)
    rf = pickle.load(f)

@app.post('/score')
async def score(client: ClientData):
    payload = client.model_dump()  # dict of features
    X = dv.transform([payload])
    pred = rf.predict(X)[0]
    return {"prediction": int(pred)}
```

Interactive API docs are available at /docs when the server is running.


## Repository Structure
- credit_card_fraud_detection.ipynb — training/experimentation notebook
- credit_card_fraud_detection.py — training script version
- credit_card_fraud_detection.pkl — serialized DictVectorizer and model (loaded by the API)
- main.py — FastAPI app exposing the scoring endpoint
- validate_server.py — simple client script to test the API
- pyproject.toml — project metadata and dependencies
- uv.lock — lockfile (for uv dependency manager)
- Dockerfile — container build spec
- fly.toml — config for Fly.io deployment (optional)



## Requirements
- Python 3.13+
- One of:
  - uv (recommended): https://docs.astral.sh/uv/
  - pip + virtual environment
- Model file credit_card_fraud_detection.pkl present at project root

If you need to (re)train the model, use the notebook or script and export the pickle to credit_card_fraud_detection.pkl.


## Setup

Option A: Using uv (recommended)
1) Install uv (see docs: https://docs.astral.sh/uv/)
2) Sync dependencies:
   uv sync

Option B: Using pip
1) Create and activate a virtual environment
   - Windows (PowerShell):
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
   - macOS/Linux (bash):
     python3 -m venv .venv
     source .venv/bin/activate
2) Install dependencies:
   pip install -r <(uv pip compile pyproject.toml)   # if uv is available to compile
   or simply:
   pip install fastapi uvicorn scikit-learn requests


## Run the API locally
Ensure credit_card_fraud_detection.pkl is in the project root.

Using uv:
- Start server:
  uv run uvicorn main:app --host 0.0.0.0 --port 8000

Using Python directly:
- Start server:
  python -m uvicorn main:app --host 0.0.0.0 --port 8000

The root endpoint returns a health message at http://127.0.0.1:8000/
The interactive docs are at http://127.0.0.1:8000/docs


## API
Base URL (local): http://127.0.0.1:8000

Endpoints:
- GET / — health message
- POST /score — returns classification for a single transaction payload

Request schema for /score (JSON):
- V1 .. V28: float
- Amount: float

Example request:
{
  "V1": -1.0,
  "V2": 0.5,
  "V3": -0.1,
  "V4": 0.2,
  "V5": -0.3,
  "V6": 0.4,
  "V7": -0.5,
  "V8": 0.6,
  "V9": -0.7,
  "V10": -0.8,
  "V11": 0.9,
  "V12": -1.0,
  "V13": 0.1,
  "V14": -0.2,
  "V15": -0.01,
  "V16": -0.3,
  "V17": -0.4,
  "V18": -0.5,
  "V19": 0.3,
  "V20": -0.17,
  "V21": 0.57,
  "V22": 0.18,
  "V23": -0.43,
  "V24": -0.05,
  "V25": 0.25,
  "V26": -0.65,
  "V27": -0.83,
  "V28": 0.85,
  "Amount": 10.0
}

Example curl:
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.0, "V2": 0.5, "V3": -0.1, "V4": 0.2, "V5": -0.3, "V6": 0.4, "V7": -0.5, "V8": 0.6, "V9": -0.7,
    "V10": -0.8, "V11": 0.9, "V12": -1.0, "V13": 0.1, "V14": -0.2, "V15": -0.01, "V16": -0.3, "V17": -0.4,
    "V18": -0.5, "V19": 0.3, "V20": -0.17, "V21": 0.57, "V22": 0.18, "V23": -0.43, "V24": -0.05, "V25": 0.25,
    "V26": -0.65, "V27": -0.83, "V28": 0.85, "Amount": 10.0
  }'

Example response:
{
  "prediction": 0
}
Where prediction is 0 for non-fraud and 1 for fraud.


## Quick validation
A small script is provided to validate the running server:
- Ensure the API is running on http://127.0.0.1:8000
- Run:
  uv run python validate_server.py
Or, if using a venv:
  python validate_server.py


## Docker
Build the image:
- docker build -t credit-card-fraud-api .

Run the container (exposes port 8000):
- docker run --rm -p 8000:8000 credit-card-fraud-api

After starting, access:
- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

Note: The Dockerfile expects credit_card_fraud_detection.pkl to be present during build.


## Retraining
- Use credit_card_fraud_detection.ipynb or credit_card_fraud_detection.py to train a new model.
- Export the DictVectorizer and model to credit_card_fraud_detection.pkl in the same order they are loaded in main.py (first dv, then model).


## Deployment
The repository includes a fly.toml which can be adapted for Fly.io deployments. Typical workflow:
- Ensure the API runs locally with the model file
- Build and push the image to a registry (or use Fly's builder)
- Configure environment/volumes if you plan to swap the model at runtime
- Deploy according to Fly.io docs


## Troubleshooting
- Import/Runtime errors: Ensure Python 3.13+ and dependencies are installed.
- 422 Unprocessable Entity: Verify request JSON contains all features V1..V28 and Amount as floats.
- Model not found: Ensure credit_card_fraud_detection.pkl is at project root before starting the API or building Docker image.
- Dependency resolution issues: If not using uv, install FastAPI, Uvicorn, scikit-learn, and requests with pip as shown.


## License
No license specified. Add a LICENSE file if you intend to share or open-source this project.

This is a project for detecting fraudulent credit card transactions. It provides:
- A trained model serialized to a single pickle file
- A FastAPI service exposing a /score endpoint for predictions
- Scripts/notebooks for training and quick validation
- Docker support for containerized deployment


## Dataset
The project uses a common anonymized credit card transactions dataset.
- Source (hosted copy): https://www.dropbox.com/scl/fi/hh8c3x23wlmi2ra0cdwqv/creditcard.csv?rlkey=lnttjl0nftukwelj5e1cf1zwn&st=bkf1yeot&dl=0
- Note: The file is large and not included in the repository.


## Repository Structure
- credit_card_fraud_detection.ipynb — training/experimentation notebook
- credit_card_fraud_detection.py — training script version
- credit_card_fraud_detection.pkl — serialized DictVectorizer and model (loaded by the API)
- main.py — FastAPI app exposing the scoring endpoint
- validate_server.py — simple client script to test the API
- pyproject.toml — project metadata and dependencies
- uv.lock — lockfile (for uv dependency manager)
- Dockerfile — container build spec
- fly.toml — config for Fly.io deployment (optional)
- portfolio/ — additional assets (if any)


## Requirements
- Python 3.13+
- One of:
  - uv (recommended): https://docs.astral.sh/uv/
  - pip + virtual environment
- Model file credit_card_fraud_detection.pkl present at project root

If you need to (re)train the model, use the notebook or script and export the pickle to credit_card_fraud_detection.pkl.


## Setup

Option A: Using uv (recommended)
1) Install uv (see docs: https://docs.astral.sh/uv/)
2) Sync dependencies:
   uv sync

Option B: Using pip
1) Create and activate a virtual environment
   - Windows (PowerShell):
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
   - macOS/Linux (bash):
     python3 -m venv .venv
     source .venv/bin/activate
2) Install dependencies:
   pip install -r <(uv pip compile pyproject.toml)   # if uv is available to compile
   or simply:
   pip install fastapi uvicorn scikit-learn requests


## Run the API locally
Ensure credit_card_fraud_detection.pkl is in the project root.

Using uv:
- Start server:
  uv run uvicorn main:app --host 0.0.0.0 --port 8000

Using Python directly:
- Start server:
  python -m uvicorn main:app --host 0.0.0.0 --port 8000

The root endpoint returns a health message at http://127.0.0.1:8000/
The interactive docs are at http://127.0.0.1:8000/docs


## API
Base URL (local): http://127.0.0.1:8000

Endpoints:
- GET / — health message
- POST /score — returns classification for a single transaction payload

Request schema for /score (JSON):
- V1 .. V28: float
- Amount: float

Example request:
{
  "V1": -1.0,
  "V2": 0.5,
  "V3": -0.1,
  "V4": 0.2,
  "V5": -0.3,
  "V6": 0.4,
  "V7": -0.5,
  "V8": 0.6,
  "V9": -0.7,
  "V10": -0.8,
  "V11": 0.9,
  "V12": -1.0,
  "V13": 0.1,
  "V14": -0.2,
  "V15": -0.01,
  "V16": -0.3,
  "V17": -0.4,
  "V18": -0.5,
  "V19": 0.3,
  "V20": -0.17,
  "V21": 0.57,
  "V22": 0.18,
  "V23": -0.43,
  "V24": -0.05,
  "V25": 0.25,
  "V26": -0.65,
  "V27": -0.83,
  "V28": 0.85,
  "Amount": 10.0
}

Example curl:
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.0, "V2": 0.5, "V3": -0.1, "V4": 0.2, "V5": -0.3, "V6": 0.4, "V7": -0.5, "V8": 0.6, "V9": -0.7,
    "V10": -0.8, "V11": 0.9, "V12": -1.0, "V13": 0.1, "V14": -0.2, "V15": -0.01, "V16": -0.3, "V17": -0.4,
    "V18": -0.5, "V19": 0.3, "V20": -0.17, "V21": 0.57, "V22": 0.18, "V23": -0.43, "V24": -0.05, "V25": 0.25,
    "V26": -0.65, "V27": -0.83, "V28": 0.85, "Amount": 10.0
  }'

Example response:
{
  "prediction": 0
}
Where prediction is 0 for non-fraud and 1 for fraud.


## Quick validation
A small script is provided to validate the running server:
- Ensure the API is running on http://127.0.0.1:8000
- Run:
  uv run python validate_server.py
Or, if using a venv:
  python validate_server.py


## Docker
Build the image:
- docker build -t credit-card-fraud-api .

Run the container (exposes port 8000):
- docker run --rm -p 8000:8000 credit-card-fraud-api

After starting, access:
- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

Note: The Dockerfile expects credit_card_fraud_detection.pkl to be present during build.


## Retraining
- Use credit_card_fraud_detection.ipynb or credit_card_fraud_detection.py to train a new model.
- Export the DictVectorizer and model to credit_card_fraud_detection.pkl in the same order they are loaded in main.py (first dv, then model).


## Deployment
The repository includes a fly.toml which can be adapted for Fly.io deployments. 

For this project the workflow include:
- Ensure the API runs locally with the model file

- Configure environment/volumes if you plan to swap the model at runtime
Changed --host 127.0.0.1 (for local connections) → --host 0.0.0.0 (allows external connections)
Removed  ENV PATH for the local 'app' created on docker image and .python-version copy
Simplified PATH handling (not needed with uv run)
Created fly.toml configuration
Updated main.py host binding to 0.0.0.0
changed port in 'main.py' and redeployed
Removed 'uv' and 'run' in dockerfile

- Build and push the image to a registry (or use Fly's builder)
install the Fly.io CLI. Let me install it using PowerShell:
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
flyctl deploy
&"$ENV:USERPROFILE\.fly\bin\flyctl.exe" deploy
Would you like to sign in? Do you want to send y followed by Enter to the terminal?
&"$ENV:USERPROFILE\.fly\bin\flyctl.exe" apps create credit-card-fraud-detection
&"$ENV:USERPROFILE\.fly\bin\flyctl.exe" deploy

- Deploy according to Fly.io docs

## Troubleshooting
- Import/Runtime errors: Ensure Python 3.13+ and dependencies are installed.
- 422 Unprocessable Entity: Verify request JSON contains all features V1..V28 and Amount as floats.
- Model not found: Ensure credit_card_fraud_detection.pkl is at project root before starting the API or building Docker image.
- Dependency resolution issues: If not using uv, install FastAPI, Uvicorn, scikit-learn, and requests with pip as shown.


## License
No license specified. Add a LICENSE file if you intend to share or open-source this project.
