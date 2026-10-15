"""
STEP 7: EVALUATE MODEL
=======================
⭐ CRITICAL STEP ⭐

Don't just use Accuracy!
Use multiple metrics:
- Accuracy
- Precision
- Recall (most important for fraud!)
- F1 Score
- ROC-AUC
- Confusion Matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_model_and_data():
    """Load trained model and test data"""
    try:
        model = joblib.load('fraud_detection_model.pkl')
        X_test = pd.read_csv('X_test_scaled.csv')
        y_test = pd.read_csv('y_test.csv').squeeze()
        
        print(f"\n✅ Model and data loaded successfully!")
        print(f"   X_test: {X_test.shape}")
        print(f"   y_test: {y_test.shape}")
        
        return model, X_test, y_test
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find required files!")
        print(f"   Missing: {str(e)}")
        print("   Please run: python 6_train_model.py first")
        return None, None, None


def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation
    """
    print("\n" + "="*60)
    print("STEP 7: MODEL EVALUATION")
    print("="*60)
    
    # ============================================================
    # PART 1: MAKE PREDICTIONS
    # ============================================================
    print("\n1️⃣  MAKING PREDICTIONS")
    print("-" * 60)
    
    print("\n🔮 Making predictions on test data...")
    y_pred = model.predict(X_test)  # Hard predictions (0 or 1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities
    
    print(f"✅ Predictions made for {len(y_test)} transactions")
    
    # ============================================================
    # PART 2: CONFUSION MATRIX
    # ============================================================
    print("\n2️⃣  CONFUSION MATRIX")
    print("-" * 60)
    
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    print("\nConfusion Matrix:")
    print(f"                Predicted Genuine    Predicted Fraud")
    print(f"Actually Genuine:        {tn:4d}              {fp:4d}")
    print(f"Actually Fraud:          {fn:4d}              {tp:4d}")
    
    print("\n📝 Interpretation:")
    print(f"  ✅ TN (True Negatives):  {tn}")
    print(f"     Correctly predicted genuine")
    print(f"  ⚠️  FP (False Positives): {fp}")
    print(f"     Genuine transaction wrongly flagged as fraud")
    print(f"  🚨 FN (False Negatives): {fn}")
    print(f"     Fraud transaction missed (WORST!)")
    print(f"  ✅ TP (True Positives):  {tp}")
    print(f"     Correctly detected fraud")
    
    # ============================================================
    # PART 3: CALCULATION OF METRICS
    # ============================================================
    print("\n3️⃣  CALCULATING METRICS")
    print("-" * 60)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nMetrics calculated!")
    
    # ============================================================
    # PART 4: DISPLAY ALL METRICS
    # ============================================================
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE METRICS")
    print("="*60)
    
    print(f"\n1. ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("   Overall correctness of predictions")
    print(f"   Correct predictions: {tp + tn} out of {len(y_test)}")
    print("   ⚠️  Not good for imbalanced data (can be misleading)")
    
    print(f"\n2. PRECISION: {precision:.4f} ({precision*100:.2f}%)")
    print("   Of predicted fraud, how many are actually fraud?")
    print(f"   Formula: TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.4f}")
    print("   ✅ Goal: Minimize false alarms")
    
    print(f"\n3. RECALL: {recall:.4f} ({recall*100:.2f}%)")
    print("   Of actual fraud, how many did we catch?")
    print(f"   Formula: TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.4f}")
    print("   ⭐ Most important! (Goal: Catch all fraud)")
    print(f"   Missing {fn} fraud cases out of {tp + fn}")
    
    print(f"\n4. F1 SCORE: {f1:.4f}")
    print("   Balance between Precision and Recall")
    print("   Good when you need both precision and recall")
    
    print(f"\n5. ROC-AUC: {roc_auc:.4f}")
    print("   How good is the model at ranking?")
    print("   1.0 = perfect, 0.5 = random")
    print("   Higher is better")
    
    # ============================================================
    # PART 5: CLASSIFICATION REPORT
    # ============================================================
    print("\n" + "="*60)
    print("📋 DETAILED CLASSIFICATION REPORT")
    print("="*60)
    
    print("\n" + classification_report(y_test, y_pred, 
                                      target_names=['Genuine', 'Fraud']))
    
    # ============================================================
    # PART 6: METRICS SUMMARY TABLE
    # ============================================================
    print("\n" + "="*60)
    print("📊 METRICS SUMMARY")
    print("="*60)
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        'Value': [accuracy, precision, recall, f1, roc_auc],
        'Percentage': [f"{accuracy*100:.2f}%", f"{precision*100:.2f}%", 
                       f"{recall*100:.2f}%", f"{f1*100:.2f}%", f"{roc_auc*100:.2f}%"]
    })
    
    print("\n" + metrics_df.to_string(index=False))
    
    # ============================================================
    # PART 7: INTERPRETATION
    # ============================================================
    print("\n" + "="*60)
    print("🔍 INTERPRETATION")
    print("="*60)
    
    print("\n✅ What does this mean?")
    print(f"   Accuracy: {accuracy*100:.1f}% of all predictions are correct")
    print(f"   Precision: {precision*100:.1f}% of fraud alerts are real fraud")
    print(f"   Recall: {recall*100:.1f}% of actual fraud is caught")
    print(f"   ROC-AUC: Model is {roc_auc*100:.1f}% confident in ranking")
    
    print("\n⭐ BEST FOR FRAUD DETECTION:")
    print(f"   High Recall ({recall*100:.1f}%) ✓ Catch most fraud")
    print(f"   Good Precision ({precision*100:.1f}%) ✓ Minimize false alarms")
    print(f"   Good ROC-AUC ({roc_auc*100:.1f}%) ✓ Reliable ranking")
    
    # ============================================================
    # PART 8: CREATE VISUALIZATIONS
    # ============================================================
    print("\n" + "="*60)
    print("📈 CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Genuine', 'Fraud'],
                yticklabels=['Genuine', 'Fraud'])
    axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xlabel('Predicted')
    
    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC={roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    axes[0, 1].fill_between(fpr, tpr, alpha=0.2)
    axes[0, 1].set_title('ROC Curve', fontweight='bold')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Metrics Comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    values = [accuracy, precision, recall, f1, roc_auc]
    colors = ['green' if v > 0.8 else 'orange' if v > 0.7 else 'red' for v in values]
    axes[1, 0].bar(metrics, values, color=colors, alpha=0.7)
    axes[1, 0].set_title('Metrics Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(values):
        axes[1, 0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Plot 4: Prediction Distribution
    axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, label='Genuine', alpha=0.7, color='green')
    axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, label='Fraud', alpha=0.7, color='red')
    axes[1, 1].set_title('Fraud Probability Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Predicted Fraud Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=100, bbox_inches='tight')
    print("\n✅ Visualizations saved as 'model_evaluation.png'")
    plt.close()
    
    # ============================================================
    # PART 9: RECOMMENDATIONS
    # ============================================================
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    if recall < 0.8:
        print("\n⚠️  Low Recall! Missing too much fraud")
        print("   → Retrain with different parameters")
        print("   → Lower fraud probability threshold")
        print("   → Collect more fraud examples")
    
    if precision < 0.7:
        print("\n⚠️  Low Precision! Too many false alarms")
        print("   → Need to balance recall vs precision")
        print("   → Adjust prediction threshold")
        print("   → Add more features")
    
    if roc_auc < 0.8:
        print("\n⚠️  Low ROC-AUC! Model not discriminating well")
        print("   → Model needs improvement")
        print("   → Try different algorithm (XGBoost, etc.)")
        print("   → Feature engineering needed")
    
    if recall >= 0.8 and precision >= 0.7:
        print("\n✅ Model is performing well!")
        print("   → Ready for deployment")
        print("   → Good recall: catches most fraud")
        print("   → Good precision: reasonable false alarms")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


if __name__ == "__main__":
    """
    Run this file to evaluate the model
    Command: python 7_evaluate_model.py
    """
    print("\n🏦 CREDIT CARD FRAUD DETECTION PROJECT")
    print("="*60)
    
    # Load model and data
    model, X_test, y_test = load_model_and_data()
    
    if model is not None:
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        print("\n" + "="*60)
        print("✅ STEP 7 COMPLETE!")
        print("="*60)
        print("\n📝 Next Step:")
        print("   Run: python 8_make_predictions.py")
        print("\n")
    else:
        print("\n❌ Cannot proceed without model!")