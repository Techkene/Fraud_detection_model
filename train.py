#!/usr/bin/env python
"""
train.py - Training script for Fraud Detection Model

This script loads the synthetic transaction data, preprocesses it,
trains an XGBoost classifier, and saves the model to a pickle file.

Usage:
    python train.py

Output:
    xgb_model.pkl - Trained XGBoost model
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'dataset.csv'
MODEL_PATH = 'xgb_model.pkl'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from CSV file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess the data for model training.
    
    - Handle missing values
    - Encode categorical variables
    - Split features and target
    - Apply SMOTE for class imbalance
    """
    print("\nPreprocessing data...")
    
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Drop unnecessary columns if they exist
    cols_to_drop = ['account_id', 'receiver_account_id', 'timestamp', 'transaction_id']
    for col in cols_to_drop:
        if col in data.columns:
            data = data.drop(columns=[col])
    
    # Handle missing values
    data = data.dropna()
    
    # Identify target column (commonly named 'is_fraud', 'fraud', 'label', or 'target')
    target_col = None
    for col in ['is_fraud', 'fraud', 'label', 'target', 'Outcome']:
        if col in data.columns:
            target_col = col
            break
    
    if target_col is None:
        # Use the last column as target
        target_col = data.columns[-1]
        print(f"Using '{target_col}' as target column")
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Encode categorical columns
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Ensure target is numeric
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{pd.Series(y).value_counts()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Apply SMOTE to handle class imbalance
    print("\nApplying SMOTE for class balancing...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Training set shape after SMOTE: {X_train_balanced.shape}")
    print(f"Balanced target distribution:\n{pd.Series(y_train_balanced).value_counts()}")
    
    return X_train_balanced, X_test, y_train_balanced, y_test, X.columns.tolist()


def train_model(X_train, y_train) -> XGBClassifier:
    """Train an XGBoost classifier."""
    print("\nTraining XGBoost model...")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy


def save_model(model, filepath: str):
    """Save the trained model to a pickle file."""
    print(f"\nSaving model to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully!")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Preprocess
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Train
    model = train_model(X_train, y_train)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Save
    save_model(model, MODEL_PATH)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nTo run the prediction service:")
    print(f"  streamlit run predict.py")


if __name__ == "__main__":
    main()
