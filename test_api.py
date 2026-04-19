import json
import os
import requests
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.expanduser("~/hw4-mlops")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")

with open(os.path.join(MODEL_DIR, "sample_input.json")) as f:
    sample = json.load(f)

X_test = pd.read_csv(os.path.join(MODEL_DIR, "X_test.csv"))

def print_header(title):
    print("=" * 60)
    print(title)
    print("=" * 60)


# TEST 1: Health
def test_health():
    print_header("TEST 1: Health Check (GET /health)")

    response = requests.get("http://localhost:5050/health")
    print(f"Status Code: {response.status_code}")

    data = response.json()
    print(json.dumps(data, indent=2))

    assert response.status_code == 200
    assert "status" in data
    assert "model" in data

    print("PASS")


# TEST 2: Single prediction
def test_single_prediction():
    print_header("TEST 2: Single Prediction (POST /predict)")

    print("Sending sample input...")
    response = requests.post("http://localhost:5050/predict", json=sample)

    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2))

    assert response.status_code == 200
    assert "prediction" in data
    assert "probability" in data
    assert "label" in data
    assert data["label"] in ("positive", "negative")

    print("PASS")


# TEST 3: Batch prediction
def test_batch_prediction():
    print_header("TEST 3: Batch Prediction (POST /predict/batch)")

    batch_data = []
    for i in range(5):
        row = X_test.iloc[i].to_dict()
        row = {
            k: int(v) if isinstance(v, (np.integer,))
            else float(v) if isinstance(v, (np.floating,))
            else str(v)
            for k, v in row.items()
        }
        batch_data.append(row)

    print(f"Sending batch of {len(batch_data)} records...")
    response = requests.post("http://localhost:5050/predict/batch", json=batch_data)

    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2))

    assert response.status_code == 200
    assert "predictions" in data
    assert "count" in data
    assert data["count"] == 5
    assert len(data["predictions"]) == 5
    # Check labels are correct values
    for pred in data["predictions"]:
        assert pred["label"] in ("positive", "negative")

    print("PASS")


# TEST 4: Missing field
def test_missing_field():
    print_header("TEST 4: Missing Required Field")

    bad_input = sample.copy()
    removed_key = list(bad_input.keys())[0]
    del bad_input[removed_key]

    print(f"Removed field: {removed_key}")
    response = requests.post("http://localhost:5050/predict", json=bad_input)

    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2))

    assert response.status_code == 400
    assert "error" in data
    assert "details" in data

    print("PASS")


# TEST 5: Invalid numeric type
def test_invalid_type():
    print_header("TEST 5: Invalid Type (freight_value = string)")

    bad_input = sample.copy()
    # freight_value is a required numeric field — passing a string must trigger 400
    bad_input["freight_value"] = "not_a_number"

    print("Corrupted field: freight_value")
    response = requests.post("http://localhost:5050/predict", json=bad_input)

    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2))

    assert response.status_code == 400
    assert "error" in data
    assert "details" in data

    print("PASS")


if __name__ == "__main__":
    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_missing_field()
        test_invalid_type()

        print("=" * 60)
        print("ALL 5 TESTS PASSED")
        print("=" * 60)

    except AssertionError as e:
        print(f"TEST FAILED: {e}")