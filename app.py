import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# ============================================================
# Initialize app and load model
# ============================================================
app = Flask(__name__)

MODEL_PATH  = os.environ.get('MODEL_PATH',  'model/model.pkl')
SCHEMA_PATH = os.environ.get('SCHEMA_PATH', 'model/schema.json')

print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print(f"Model loaded: {type(model).__name__}")

with open(SCHEMA_PATH) as f:
    schema = json.load(f)

REQUIRED_FEATURES     = schema['all_features']
NUMERIC_FEATURES      = schema['numeric_features']
CATEGORICAL_FEATURES  = schema['categorical_features']

# ── Validation rule sets ───────────────────────────────────────────────────
# Must be strictly > 0
POSITIVE_ONLY_FIELDS = {
    "product_weight_g",
    "product_volume",
    "total_order_items",
    "seller_orders_count",
}

# Must be >= 0
NON_NEGATIVE_FIELDS = {
    "freight_value",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "customer_zip_code_prefix",
    "seller_zip_code_prefix",
    "log_price",
}

# Must be 0 or 1
BINARY_FIELDS = {
    "is_repeat_customer",
    "seller_state_match",
    "is_delivered",
}

# Can be any value — including negative (early delivery = negative delta)
# delivery_vs_estimated  →  no range constraint needed


# ============================================================
# Helper: Input Validation
# ============================================================
def validate_input(data):
    """Validate a single prediction input. Returns (is_valid, error_dict)."""
    errors = {}

    # 1. Missing fields
    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        errors['missing_fields'] = missing

    # 2. Numeric fields must be castable to float
    for field in NUMERIC_FEATURES:
        if field in data:
            try:
                float(data[field])
            except (ValueError, TypeError):
                errors[field] = f'expected numeric, got {type(data[field]).__name__}'

    # 3. Range checks (only if field passed the numeric check above)
    for field in POSITIVE_ONLY_FIELDS:
        if field in data and field not in errors:
            if float(data[field]) <= 0:
                errors[field] = 'must be a positive number (> 0)'

    for field in NON_NEGATIVE_FIELDS:
        if field in data and field not in errors:
            if float(data[field]) < 0:
                errors[field] = 'must be a non-negative number (>= 0)'

    for field in BINARY_FIELDS:
        if field in data and field not in errors:
            if int(float(data[field])) not in (0, 1):
                errors[field] = 'must be 0 or 1'

    # 4. Categorical fields must be strings
    for field in CATEGORICAL_FEATURES:
        if field in data:
            if not isinstance(data[field], str):
                errors[field] = f'expected string category, got {type(data[field]).__name__}'

    return len(errors) == 0, errors


# ============================================================
# Endpoints
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check — is the API alive and model loaded?"""
    if model is None:
        return jsonify({'status': 'error', 'message': 'model not loaded'}), 503
    return jsonify({'status': 'healthy', 'model': 'loaded'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Expected a JSON object'}), 400
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400

    is_valid, errors = validate_input(data)
    if not is_valid:
        return jsonify({'error': 'Invalid input', 'details': errors}), 400

    try:
        df   = pd.DataFrame([data])[REQUIRED_FEATURES]
        pred  = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]

        return jsonify({
            'prediction':  int(pred),
            'probability': round(float(proba), 4),
            'label':       'positive' if pred == 1 else 'negative'
        }), 200
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint. Accepts a JSON array."""
    data = request.get_json(silent=True)
    if not isinstance(data, list):
        return jsonify({'error': 'Expected a JSON array'}), 400
    if len(data) == 0:
        return jsonify({'error': 'Empty array'}), 400
    if len(data) > 100:
        return jsonify({'error': f'Max 100 records per batch, got {len(data)}'}), 400

    all_errors = {}
    for i, record in enumerate(data):
        is_valid, errors = validate_input(record)
        if not is_valid:
            all_errors[f'record_{i}'] = errors

    if all_errors:
        return jsonify({'error': 'Invalid inputs', 'details': all_errors}), 400

    try:
        df     = pd.DataFrame(data)[REQUIRED_FEATURES]
        preds  = model.predict(df)
        probas = model.predict_proba(df)[:, 1]

        results = [{
            'prediction':  int(p),
            'probability': round(float(pr), 4),
            'label':       'positive' if p == 1 else 'negative'
        } for p, pr in zip(preds, probas)]

        return jsonify({'predictions': results, 'count': len(results)}), 200
    except Exception as e:
        return jsonify({'error': 'Batch prediction failed', 'message': str(e)}), 500


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)