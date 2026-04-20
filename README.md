# hw4-mlops

## Project Overview

This project is part of a Data Science MLOps course (HW4). The goal was to take a trained machine learning model from HW2, wrap it in a production-ready API, containerize it, and deploy it to the cloud.

**What the model does:** Given features about an order (delivery time, freight cost, product details, seller info, payment type), it predicts whether the customer will leave a positive review (`1`) or not (`0`).

**Model deployed:** Tuned Random Forest Classifier, trained on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

---

## Live URL

```
https://hw4-mlops-5csb.onrender.com
```

Health check: [https://hw4-mlops-5csb.onrender.com/health](https://hw4-mlops-5csb.onrender.com/health)

---

## API Documentation

### `GET /health`

Check that the API is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded"
}
```

---

### `POST /predict`

Make a single prediction.

**Request body:** JSON object with all required features.

**Example request:**
```bash
curl -X POST https://hw4-mlops-5csb.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_zip_code_prefix": "48120",
    "customer_city": "pojuca",
    "freight_value": 29.54,
    "product_name_lenght": 44.0,
    "product_description_lenght": 487.0,
    "product_photos_qty": 1.0,
    "product_weight_g": 2800.0,
    "product_category_name_english": "bed_bath_table",
    "seller_zip_code_prefix": 14940.0,
    "seller_city": "ibitinga",
    "seller_state": "SP",
    "payment_type": "credit_card",
    "delivery_days": 8.0,
    "delivery_vs_estimated": -23.0,
    "is_repeat_customer": 1,
    "seller_state_match": 0,
    "product_volume": 8400.0,
    "total_order_items": 2.0,
    "log_price": 4.787,
    "seller_orders_count": 116.0,
    "is_delivered": 1
  }'
```

**Example response:**
```json
{
  "prediction": 1,
  "probability": 0.8731,
  "label": "positive"
}
```

---

### `POST /predict/batch`

Make predictions for multiple orders at once (max 100 records).

**Request body:** JSON array of objects, each with the same fields as `/predict`.

**Example request:**
```bash
curl -X POST https://hw4-mlops-5csb.onrender.com/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{...record 1...}, {...record 2...}]'
```

**Example response:**
```json
{
  "predictions": [
    {"prediction": 1, "probability": 0.8731, "label": "positive"},
    {"prediction": 0, "probability": 0.3214, "label": "negative"}
  ],
  "count": 2
}
```

---

## Input Schema

All features are required. Missing fields will return a `400` error.

| Feature | Type | Valid Values / Range |
|---|---|---|
| `customer_zip_code_prefix` | int | ≥ 0 |
| `customer_city` | string | any city name |
| `freight_value` | float | ≥ 0 |
| `product_name_lenght` | float | ≥ 0 |
| `product_description_lenght` | float | ≥ 0 |
| `product_photos_qty` | float | ≥ 0 |
| `product_weight_g` | float | > 0 |
| `product_category_name_english` | string | e.g. `"bed_bath_table"`, `"electronics"` |
| `seller_zip_code_prefix` | float | ≥ 0 |
| `seller_city` | string | any city name |
| `seller_state` | string | Brazilian state code, e.g. `"SP"`, `"RJ"` |
| `payment_type` | string | `"credit_card"`, `"boleto"`, `"voucher"`, `"debit_card"` |
| `delivery_days` | float | > 0 |
| `delivery_vs_estimated` | float | any (negative = early, positive = late) |
| `is_repeat_customer` | int | `0` or `1` |
| `seller_state_match` | int | `0` or `1` |
| `product_volume` | float | > 0 |
| `total_order_items` | float | > 0 |
| `log_price` | float | ≥ 0 |
| `seller_orders_count` | float | > 0 |
| `is_delivered` | int | `0` or `1` |

---

## Local Setup

### Option 1 - Run without Docker

**Requirements:** Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/argareeva/hw4-mlops.git
cd hw4-mlops

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

API will be available at `http://localhost:5050`

---

### Option 2 — Run with Docker

**Requirements:** Docker Desktop

```bash
# 1. Clone the repo
git clone https://github.com/argareeva/hw4-mlops.git
cd hw4-mlops

# 2. Build the image
docker build -t hw4-api .

# 3. Run the container
docker run -d --name hw4-api -p 5050:5050 hw4-api

# 4. Check it's working
curl http://localhost:5050/health
```

To stop the container:
```bash
docker stop hw4-api && docker rm hw4-api
```

---

## Model Information

| Property | Value |
|---|---|
| Model type | Random Forest Classifier (tuned) |
| Training data | Olist Brazilian E-Commerce dataset |
| Train/test split | 80% / 20%, stratified |
| Hyperparameters | `n_estimators=50`, `max_depth=5`, `min_samples_split=2` |
| **Accuracy** | 0.650 |
| **F1 Score** | 0.752 |
| **Recall** | 0.855 |
| **Precision** | 0.671 |

**Known limitations:**
- The model was trained on a relatively small sample (500 records from the full dataset) for the purposes of this assignment, which limits generalization
- Features like `delivery_vs_estimated` require knowing the estimated delivery date upfront, which may not always be available in real time
- The model does not use review text — a BERT-based sentiment model tested in Part 1 significantly outperformed it (F1 = 0.84 vs 0.75)

---
