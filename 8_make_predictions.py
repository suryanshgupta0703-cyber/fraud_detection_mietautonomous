"""
STEP 8: MAKE PREDICTIONS
========================
Uses trained model to predict fraud on new transactions

This is where the model is actually used!
Input: Transaction details
Output: Fraud or Genuine prediction
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_model_and_scaler():
    """Load trained model and scaler"""
    try:
        model = joblib.load('fraud_detection_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        print(f"\n✅ Model and scaler loaded successfully!")
        
        return model, scaler
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find required files!")
        print(f"   Missing: {str(e)}")
        print("   Please run: python 6_train_model.py first")
        return None, None


def predict_single_transaction(model, scaler, transaction_data):
    """
    Predict fraud for a single transaction
    
    transaction_data should be a list or dict with:
    - amount: ₹ amount
    - time: hour (0-23)
    - merchant_type: 1-5
    - card_type: 0/1/2
    - international: 0/1
    """
    
    # Convert to DataFrame for proper column alignment
    if isinstance(transaction_data, list):
        transaction_df = pd.DataFrame([transaction_data], 
                                     columns=['amount', 'time', 'merchant_type', 'card_type', 'international'])
    else:
        transaction_df = pd.DataFrame([transaction_data])
    
    # Scale the transaction
    transaction_scaled = scaler.transform(transaction_df)
    
    # Make prediction
    prediction = model.predict(transaction_scaled)[0]  # 0 or 1
    probability = model.predict_proba(transaction_scaled)[0]  # [prob_genuine, prob_fraud]
    
    return prediction, probability


def run_prediction_examples():
    """
    Run multiple prediction examples
    """
    print("\n" + "="*60)
    print("STEP 8: MAKING PREDICTIONS")
    print("="*60)
    
    model, scaler = load_model_and_scaler()
    
    if model is None:
        return
    
    print("\n" + "="*60)
    print("📝 TRANSACTION FORMAT")
    print("="*60)
    
    print("\nEach transaction has 5 features:")
    print("  1. amount: ₹ transaction amount (100-10000)")
    print("  2. time: hour of day (0-23)")
    print("  3. merchant_type: 1-5")
    print("  4. card_type: 0=Debit, 1=Credit, 2=Prepaid")
    print("  5. international: 0=Domestic, 1=International")
    
    # ============================================================
    # EXAMPLE 1: Normal transaction (likely genuine)
    # ============================================================
    print("\n" + "="*60)
    print("📊 EXAMPLE 1: Normal Genuine Transaction")
    print("="*60)
    
    transaction1 = [3000, 10, 2, 1, 0]  # ₹3000, 10 AM, Merchant 2, Credit, Domestic
    
    print("\nTransaction details:")
    print(f"  Amount: ₹{transaction1[0]}")
    print(f"  Time: {transaction1[1]:02d}:00")
    print(f"  Merchant Type: {transaction1[2]}")
    print(f"  Card Type: {'Debit' if transaction1[3]==0 else 'Credit' if transaction1[3]==1 else 'Prepaid'}")
    print(f"  International: {'Yes' if transaction1[4]==1 else 'No'}")
    
    pred1, prob1 = predict_single_transaction(model, scaler, transaction1)
    
    print("\n🔮 PREDICTION:")
    print(f"  Prediction: {['✓ GENUINE', '🚨 FRAUD'][pred1]}")
    print(f"  Genuine probability: {prob1[0]:.2%}")
    print(f"  Fraud probability: {prob1[1]:.2%}")
    print(f"  Confidence: {max(prob1):.2%}")
    
    # ============================================================
    # EXAMPLE 2: Suspicious transaction (likely fraud)
    # ============================================================
    print("\n" + "="*60)
    print("📊 EXAMPLE 2: Suspicious Fraudulent Transaction")
    print("="*60)
    
    transaction2 = [9000, 3, 1, 0, 1]  # ₹9000, 3 AM, Merchant 1, Debit, International
    
    print("\nTransaction details:")
    print(f"  Amount: ₹{transaction2[0]}")
    print(f"  Time: {transaction2[1]:02d}:00")
    print(f"  Merchant Type: {transaction2[2]}")
    print(f"  Card Type: {'Debit' if transaction2[3]==0 else 'Credit' if transaction2[3]==1 else 'Prepaid'}")
    print(f"  International: {'Yes' if transaction2[4]==1 else 'No'}")
    
    pred2, prob2 = predict_single_transaction(model, scaler, transaction2)
    
    print("\n🔮 PREDICTION:")
    print(f"  Prediction: {['✓ GENUINE', '🚨 FRAUD'][pred2]}")
    print(f"  Genuine probability: {prob2[0]:.2%}")
    print(f"  Fraud probability: {prob2[1]:.2%}")
    print(f"  Confidence: {max(prob2):.2%}")
    
    print("\n🔍 SUSPICIOUS FACTORS:")
    if transaction2[0] > 5000:
        print("  • High amount (₹9000)")
    if transaction2[1] in [0, 1, 2, 3, 23]:
        print(f"  • Unusual time (3 AM)")
    if transaction2[4] == 1:
        print("  • International transaction")
    
    # ============================================================
    # EXAMPLE 3: Borderline transaction (medium risk)
    # ============================================================
    print("\n" + "="*60)
    print("📊 EXAMPLE 3: Medium Risk Transaction")
    print("="*60)
    
    transaction3 = [5500, 15, 3, 1, 1]  # ₹5500, 3 PM, Merchant 3, Credit, International
    
    print("\nTransaction details:")
    print(f"  Amount: ₹{transaction3[0]}")
    print(f"  Time: {transaction3[1]:02d}:00")
    print(f"  Merchant Type: {transaction3[2]}")
    print(f"  Card Type: {'Debit' if transaction3[3]==0 else 'Credit' if transaction3[3]==1 else 'Prepaid'}")
    print(f"  International: {'Yes' if transaction3[4]==1 else 'No'}")
    
    pred3, prob3 = predict_single_transaction(model, scaler, transaction3)
    
    print("\n🔮 PREDICTION:")
    print(f"  Prediction: {['✓ GENUINE', '🚨 FRAUD'][pred3]}")
    print(f"  Genuine probability: {prob3[0]:.2%}")
    print(f"  Fraud probability: {prob3[1]:.2%}")
    print(f"  Confidence: {max(prob3):.2%}")
    
    # ============================================================
    # BATCH PREDICTIONS (from test data)
    # ============================================================
    print("\n" + "="*60)
    print("📊 BATCH PREDICTIONS (Sample from Test Data)")
    print("="*60)
    
    try:
        X_test = pd.read_csv('X_test_scaled.csv')
        y_test = pd.read_csv('y_test.csv').squeeze()
        
        # Get first 5 test samples
        X_sample = X_test.head(5)
        y_sample = y_test.head(5)
        
        print(f"\nMaking predictions on first 5 test samples...")
        
        predictions = model.predict(X_sample.values)
        probabilities = model.predict_proba(X_sample.values)[:, 1]
        
        print("\n" + "-" * 60)
        for i in range(len(X_sample)):
            actual = ['Genuine', 'Fraud'][y_sample.iloc[i]]
            predicted = ['Genuine', 'Fraud'][predictions[i]]
            prob_fraud = probabilities[i]
            match = "✓" if predictions[i] == y_sample.iloc[i] else "✗"
            
            print(f"\nTransaction {i+1}: {match}")
            print(f"  Actual: {actual}")
            print(f"  Predicted: {predicted}")
            print(f"  Fraud Probability: {prob_fraud:.2%}")
    
    except FileNotFoundError:
        print("(Test data not available for batch predictions)")
    
    # ============================================================
    # HOW TO USE IN PRODUCTION
    # ============================================================
    print("\n" + "="*60)
    print("🚀 HOW TO USE THIS IN PRODUCTION")
    print("="*60)
    
    print("\n✅ Step 1: Load model and scaler once")
    print("   model = joblib.load('fraud_detection_model.pkl')")
    print("   scaler = joblib.load('scaler.pkl')")
    
    print("\n✅ Step 2: For each transaction, prepare data")
    print("   transaction = [amount, time, merchant, card_type, international]")
    
    print("\n✅ Step 3: Scale and predict")
    print("   transaction_scaled = scaler.transform([transaction])")
    print("   prediction = model.predict(transaction_scaled)[0]")
    print("   probability = model.predict_proba(transaction_scaled)[0]")
    
    print("\n✅ Step 4: Take action")
    print("   if probability[1] > 0.5:")
    print("       flag_as_fraud()")
    print("   else:")
    print("       approve_transaction()")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    print("\n✅ Model can now make predictions!")
    print("   • Real-time fraud detection")
    print("   • Uses probability scores")
    print("   • Adaptable thresholds")
    print("   • Ready for deployment")


if __name__ == "__main__":
    """
    Run this file to make predictions
    Command: python 8_make_predictions.py
    """
    print("\n🏦 CREDIT CARD FRAUD DETECTION PROJECT")
    print("="*60)
    
    # Run prediction examples
    run_prediction_examples()
    
    print("\n" + "="*60)
    print("✅ STEP 8 COMPLETE!")
    print("="*60)
    print("\n📝 Next Step:")
    print("   Run: python 9_streamlit_app.py")
    print("\n")