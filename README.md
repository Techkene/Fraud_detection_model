# Fraud Detection Model

## Problem Description

Financial fraud costs institutions billions of dollars annually and affects millions of customers worldwide. This project develops a **machine learning-based fraud detection system** that analyzes transaction patterns to identify potentially fraudulent activities in real-time.

The solution uses an **XGBoost classifier** trained on transaction data to predict whether a given transaction is fraudulent or legitimate. The model considers various features including transaction amount, account age, transaction frequency, and behavioral patterns.

### Key Features:
- **REST API** via Flask for easy integration
- **High accuracy** XGBoost model with SMOTE for class balancing
- **Cloud deployment** on Render

---

## Dataset

The dataset (`dataset.csv`) contains transaction data with the following key features:

| Feature | Description |
|---------|-------------|
| `account_id` | Unique identifier for the account |
| `transaction_amount` | Amount of the transaction |
| `account_age_days` | Age of the account in days |
| `daily_transaction_amount` | Total transaction amount for the day |
| `total_daily_transactions` | Number of transactions for the day |
| `transaction_frequency` | Frequency of transactions |
| `account_type_personal` | Whether account is personal (1) or business (0) |
| `payment_type_debit` | Whether payment is debit (1) or credit (0) |
| `is_fraud` | Target variable: 1 = Fraud, 0 = Legitimate |

---

## Project Structure

```
Fraud_detection_model/
├── README.md              # Project documentation
├── notebook.ipynb         # EDA and model development
├── train.py               # Training script
├── predict.py             # Flask API service
├── xgb_model_2.pkl        # Trained XGBoost model
├── dataset.csv            # Training dataset
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
├── render.yaml            # Render deployment config
└── .gitignore             # Git ignore rules
```

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Techkene/Fraud_detection_model.git
cd Fraud_detection_model
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional)
```bash
python train.py
```

### 5. Run the API
```bash
python predict.py
```
Access at `http://localhost:5000`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page with API documentation |
| GET | `/health` | Health check |
| POST | `/predict` | Make a fraud prediction |

### Example Request

```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 5000.00,
    "account_age_days": 365,
    "daily_transaction_amount": 10000.00,
    "total_daily_transactions": 5,
    "transaction_frequency": 2.5
  }'
```

### Example Response

```json
{
  "prediction": 0,
  "result": "SAFE",
  "message": "Transaction appears to be legitimate"
}
```

---

## Docker Deployment

```bash
# Build
docker build -t fraud-detection .

# Run
docker run -p 5000:5000 fraud-detection
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~95% |
| Precision (Fraud) | ~93% |
| Recall (Fraud) | ~94% |
| F1-Score | ~93% |

---

## Author

**Techkene** - Data Scientist / ML Engineer

## License

This project is for educational purposes as part of the ML Zoomcamp course.
