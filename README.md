# Fraud Detection Model

## Problem Description

Financial fraud costs institutions billions of dollars annually and affects millions of customers worldwide. This project develops a **machine learning-based fraud detection system** that analyzes transaction patterns to identify potentially fraudulent activities in real-time.

The solution uses an **XGBoost classifier** trained on synthetic transaction data to predict whether a given transaction is fraudulent or legitimate. The model considers various features including transaction amount, account age, transaction frequency, and behavioral patterns.

### Key Features:
- **Real-time prediction** via Streamlit web interface
- **High accuracy** XGBoost model with SMOTE for class balancing
- **Easy deployment** with Docker containerization

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
├── predict.py             # Streamlit web service
├── xgb_model_2.pkl        # Trained XGBoost model
├── dataset.csv            # Training dataset
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
└── .gitignore             # Git ignore rules
```

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Techkene/Fraud_detection_model.git
cd Fraud_detection_model
```

### 2. Create Virtual Environment (Recommended)
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
If you want to retrain the model:
```bash
python train.py
```

### 5. Run the Prediction Service
```bash
streamlit run predict.py
```
The app will open at `http://localhost:8501`

---

## Docker Deployment

### Build the Docker Image
```bash
docker build -t fraud-detection .
```

### Run the Container
```bash
docker run -p 8501:8501 fraud-detection
```

Access the app at `http://localhost:8501`

---

## Cloud Deployment

The model is deployed on **Streamlit Cloud**:

🔗 **Live Demo**: [Add your Streamlit Cloud URL here]

### Deploying to Streamlit Cloud:
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `predict.py` as the main file
5. Deploy!

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~95% |
| Precision (Fraud) | ~93% |
| Recall (Fraud) | ~94% |
| F1-Score | ~93% |

---

## Usage Example

To make a prediction, upload a `.txt` file containing transaction data in JSON format:

```json
{
    "account_id": 12345,
    "receiver_account_id": 67890,
    "transaction_amount": 5000.00,
    "account_age_days": 365,
    "daily_transaction_amount": 10000.00,
    "total_daily_transactions": 5,
    "transaction_frequency": 2.5,
    "transaction_frequency_same_account": 1,
    "account_type_personal": true,
    "payment_type_debit": true,
    "transaction_type_bank_transfer": true,
    "transaction_type_Deposit": false,
    "transaction_type_sporty": false
}
```

---

## Author

**Techkene** - Data Scientist / ML Engineer

---

## License

This project is for educational purposes as part of the ML Zoomcamp course.
