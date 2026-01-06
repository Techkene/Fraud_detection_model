"""
predict.py - Flask API for Fraud Detection Model (Vercel Compatible)

This script loads the trained XGBoost model and serves predictions
via a REST API endpoint.

Usage:
    Local: python predict.py
    Vercel: Deployed automatically
"""

from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'xgb_model_2.pkl')

model = None
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Warning: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

# HTML template for home page
HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>KENAM - Fraud Detection API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 40px 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        .header { 
            background: linear-gradient(90deg, #0066ff, #00ccff);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 102, 255, 0.3);
        }
        h1 { font-size: 2.5rem; margin-bottom: 10px; }
        .subtitle { color: rgba(255,255,255,0.8); }
        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        h2 { color: #00ccff; margin-bottom: 15px; }
        code {
            background: rgba(0,0,0,0.3);
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
        }
        pre {
            background: rgba(0,0,0,0.4);
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            margin: 15px 0;
            font-size: 0.85rem;
        }
        .endpoint { color: #00ff88; }
        .method { 
            display: inline-block;
            padding: 4px 12px;
            border-radius: 5px;
            font-weight: bold;
            margin-right: 10px;
        }
        .get { background: #00aa55; }
        .post { background: #0088ff; }
        .status { 
            display: inline-block;
            padding: 8px 16px;
            background: #00ff88;
            color: #000;
            border-radius: 20px;
            font-weight: bold;
        }
        .status.error { background: #ff4444; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ KENAM</h1>
            <p class="subtitle">Fraud Prevention for a Safer Fintech Future</p>
        </div>
        
        <div class="card">
            <h2>API Status</h2>
            <p>Model Status: <span class="status {{ 'error' if not model_loaded else '' }}">{{ '✓ Loaded' if model_loaded else '✗ Not Loaded' }}</span></p>
        </div>
        
        <div class="card">
            <h2>Endpoints</h2>
            <p style="margin-bottom: 15px;">
                <span class="method get">GET</span>
                <code class="endpoint">/</code> - This page
            </p>
            <p style="margin-bottom: 15px;">
                <span class="method get">GET</span>
                <code class="endpoint">/health</code> - Health check
            </p>
            <p>
                <span class="method post">POST</span>
                <code class="endpoint">/predict</code> - Make a prediction
            </p>
        </div>
        
        <div class="card">
            <h2>Example Request</h2>
            <pre>curl -X POST {{ host }}/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "transaction_amount": 5000.00,
    "account_age_days": 365,
    "daily_transaction_amount": 10000.00,
    "total_daily_transactions": 5,
    "transaction_frequency": 2.5
  }'</pre>
        </div>
        
        <div class="card">
            <h2>Example Response</h2>
            <pre>{
  "prediction": 0,
  "result": "SAFE",
  "message": "Transaction appears to be legitimate"
}</pre>
        </div>
    </div>
</body>
</html>
"""


@app.route('/')
def home():
    """Home page with API documentation."""
    host = request.host_url.rstrip('/')
    return render_template_string(HOME_PAGE, model_loaded=(model is not None), host=host)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a fraud prediction.
    
    Expects JSON body with transaction features.
    Returns prediction (0=safe, 1=fraud) and result message.
    """
    if model is None:
        return jsonify({
            "error": "Model not loaded. Please ensure xgb_model_2.pkl exists."
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided. Send transaction data in request body."
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Handle boolean fields
        bool_columns = [
            'account_type_personal', 'payment_type_debit',
            'transaction_type_bank_transfer', 'transaction_type_sporty',
            'transaction_type_Deposit'
        ]
        
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 1 if x in [True, 'True', 'true', 1, '1'] else 0)
        
        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN with 0
        df = df.fillna(0)
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Prepare response
        if prediction == 0:
            result = {
                "prediction": int(prediction),
                "result": "SAFE",
                "message": "Transaction appears to be legitimate"
            }
        else:
            result = {
                "prediction": int(prediction),
                "result": "FRAUD",
                "message": "⚠️ ALERT! Transaction appears suspicious"
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


# Vercel requires the app to be named 'app'
# This is automatically detected by @vercel/python

# For local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
