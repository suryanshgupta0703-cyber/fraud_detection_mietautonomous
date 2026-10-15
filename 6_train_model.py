"""
STEP 6: TRAIN MODEL
===================
Trains Random Forest Classifier

Why Random Forest?
- Handles non-linear patterns
- Fast training
- Good for fraud detection
- Provides feature importance
- Robust to outliers
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_scaled_data():
    """Load the scaled train-test data from Step 5"""
    try:
        X_train_scaled = pd.read_csv('X_train_scaled.csv')
        y_train = pd.read_csv('y_train_balanced.csv').squeeze()
        X_test_scaled = pd.read_csv('X_test_scaled.csv')
        y_test = pd.read_csv('y_test.csv').squeeze()
        
        print(f"\n✅ Scaled data loaded successfully!")
        print(f"   X_train: {X_train_scaled.shape}")
        print(f"   y_train: {y_train.shape}")
        print(f"   X_test: {X_test_scaled.shape}")
        print(f"   y_test: {y_test.shape}")
        
        return X_train_scaled, y_train, X_test_scaled, y_test
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find required CSV files!")
        print(f"   Missing: {str(e)}")
        print("   Please run: python 5_feature_scaling.py first")
        return None, None, None, None


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest Classifier
    """
    print("\n" + "="*60)
    print("STEP 6: TRAINING RANDOM FOREST MODEL")
    print("="*60)
    
    # ============================================================
    # PART 1: WHAT IS RANDOM FOREST?
    # ============================================================
    print("\n1️⃣  RANDOM FOREST EXPLAINED")
    print("-" * 60)
    
    print("\n🌲 Random Forest creates multiple decision trees:")
    print("   • Each tree learns different patterns")
    print("   • Each tree trained on random data subset")
    print("   • Final prediction = majority vote")
    print("   • Reduces overfitting")
    print("   • Fast and reliable")
    
    # ============================================================
    # PART 2: CREATE MODEL
    # ============================================================
    print("\n2️⃣  CREATING MODEL")
    print("-" * 60)
    
    print("\n🔧 Hyperparameters:")
    print("   • n_estimators=100     (number of trees)")
    print("   • max_depth=15         (tree depth limit)")
    print("   • random_state=42      (reproducibility)")
    print("   • class_weight='balanced' (handle imbalance)")
    print("   • n_jobs=-1            (use all CPU cores)")
    
    model = RandomForestClassifier(
        n_estimators=100,           # Number of trees
        max_depth=15,               # Maximum depth of each tree
        random_state=42,            # For reproducibility
        class_weight='balanced',    # Extra weight for minority class
        n_jobs=-1                   # Use all CPU cores for speed
    )
    
    print("\n✅ Model created!")
    
    # ============================================================
    # PART 3: TRAIN MODEL
    # ============================================================
    print("\n3️⃣  TRAINING MODEL")
    print("-" * 60)
    
    print("\n🔄 Training on balanced data...")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Genuine: {(y_train == 0).sum()}")
    print(f"   Fraud: {(y_train == 1).sum()}")
    
    model.fit(X_train, y_train)
    
    print("\n✅ Model training complete!")
    
    # ============================================================
    # PART 4: TRAINING PERFORMANCE
    # ============================================================
    print("\n4️⃣  TRAINING PERFORMANCE")
    print("-" * 60)
    
    train_score = model.score(X_train, y_train)
    print(f"\nAccuracy on training data: {train_score:.4f} ({train_score*100:.2f}%)")
    
    # ============================================================
    # PART 5: TEST PERFORMANCE
    # ============================================================
    print("\n5️⃣  TEST PERFORMANCE (Quick Check)")
    print("-" * 60)
    
    test_score = model.score(X_test, y_test)
    print(f"\nAccuracy on test data: {test_score:.4f} ({test_score*100:.2f}%)")
    
    # ============================================================
    # PART 6: FEATURE IMPORTANCE
    # ============================================================
    print("\n6️⃣  FEATURE IMPORTANCE")
    print("-" * 60)
    
    print("\n📊 How important is each feature?")
    print("   (Higher = more important for predictions)")
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance Ranking:")
    for idx, row in feature_importance.iterrows():
        bar_length = int(row['importance'] * 50)
        bar = '█' * bar_length
        print(f"  {row['feature']:15s}: {bar} {row['importance']:.4f}")
    
    print("\n✅ KEY INSIGHTS:")
    top_feature = feature_importance.iloc[0]
    print(f"   Most important: {top_feature['feature']} ({top_feature['importance']:.1%})")
    print(f"   Model uses ALL features to make decisions")
    
    # ============================================================
    # PART 7: SAVE MODEL
    # ============================================================
    print("\n" + "="*60)
    print("💾 SAVING MODEL")
    print("="*60)
    
    model_filename = 'fraud_detection_model.pkl'
    joblib.dump(model, model_filename)
    
    print(f"\n✅ Model saved as '{model_filename}'")
    print("   (Will use this for predictions in Step 8)")
    
    # ============================================================
    # PART 8: MODEL SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("📋 MODEL SUMMARY")
    print("="*60)
    
    print("\n✅ Model Details:")
    print(f"   Type: Random Forest Classifier")
    print(f"   Trees: 100")
    print(f"   Max Depth: 15")
    print(f"   Training Samples: {len(X_train)}")
    print(f"   Test Samples: {len(X_test)}")
    
    print("\n✅ Performance:")
    print(f"   Training Accuracy: {train_score:.2%}")
    print(f"   Test Accuracy: {test_score:.2%}")
    print(f"   (Full evaluation in next step)")
    
    print("\n✅ Feature Importance:")
    for i, row in feature_importance.head(3).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.1%}")
    
    # ============================================================
    # PART 9: NEXT STEPS
    # ============================================================
    print("\n" + "="*60)
    print("🎯 WHAT'S NEXT?")
    print("="*60)
    
    print("\n✅ Model is trained and ready!")
    print("   • Step 7: Detailed evaluation with metrics")
    print("   • Step 8: Make predictions on new data")
    print("   • Step 9: Deploy with Streamlit web app")
    
    return model, feature_importance


if __name__ == "__main__":
    """
    Run this file to train the model
    Command: python 6_train_model.py
    """
    print("\n🏦 CREDIT CARD FRAUD DETECTION PROJECT")
    print("="*60)
    
    # Load scaled data
    X_train, y_train, X_test, y_test = load_scaled_data()
    
    if X_train is not None:
        # Train model
        model, feature_importance = train_random_forest(X_train, y_train, X_test, y_test)
        
        print("\n" + "="*60)
        print("✅ STEP 6 COMPLETE!")
        print("="*60)
        print("\n📝 Next Step:")
        print("   Run: python 7_evaluate_model.py")
        print("\n")
    else:
        print("\n❌ Cannot proceed without data!")