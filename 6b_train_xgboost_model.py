"""
STEP 6b: TRAIN XGBOOST MODEL
=============================
Alternative model training using XGBoost for comparison with Random Forest

Run with: python 6b_train_xgboost_model.py
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_xgboost_model():
    """
    Train XGBoost classifier on balanced data
    """
    print("\n" + "="*60)
    print("STEP 6b: TRAIN XGBOOST MODEL")
    print("="*60)

    # Load scaled training data
    print("\n📊 Loading training data...")
    X_train = pd.read_csv('X_train_scaled.csv')
    y_train = pd.read_csv('y_train_balanced.csv')
    X_test = pd.read_csv('X_test_scaled.csv')
    y_test = pd.read_csv('y_test.csv')

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Initialize XGBoost model
    print("\n🤖 Initializing XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    # Train model
    print("🎯 Training XGBoost model...")
    xgb_model.fit(X_train, y_train.values.ravel())

    # Make predictions
    print("🔮 Making predictions...")
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    y_proba_test = xgb_model.predict_proba(X_test)

    # Calculate metrics
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE METRICS")
    print("="*60)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, y_proba_test[:, 1])

    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Feature importance
    print("\n" + "="*60)
    print("🔍 FEATURE IMPORTANCE")
    print("="*60)

    feature_names = ['Amount', 'Time', 'Merchant', 'Card Type', 'International']
    importance = xgb_model.feature_importances_

    for name, imp in zip(feature_names, importance):
        print(f"{name}: {imp:.4f}")

    # Save model
    print("\n" + "="*60)
    print("💾 SAVING MODEL")
    print("="*60)

    joblib.dump(xgb_model, 'xgboost_model.pkl')
    print("\n✅ XGBoost model saved as 'xgboost_model.pkl'")

    # Create feature importance visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feature_names, importance, color='#059669')
    ax.set_xlabel('Importance Score')
    ax.set_title('XGBoost Feature Importance')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png', dpi=150, bbox_inches='tight')
    print("✅ Feature importance saved as 'xgboost_feature_importance.png'")

    return xgb_model


if __name__ == "__main__":
    print("\n🚀 XGBOOST MODEL TRAINING")
    print("="*60)

    model = train_xgboost_model()

    print("\n" + "="*60)
    print("✅ STEP 6b COMPLETE!")
    print("="*60)
    print("\n📝 Model saved as: xgboost_model.pkl")
    print("📊 Feature importance saved as: xgboost_feature_importance.png")
    print("\n📝 Next Step:")
    print("   Use in web app for model comparison")
    print("\n")
