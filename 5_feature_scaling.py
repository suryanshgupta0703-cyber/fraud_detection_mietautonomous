"""
STEP 5: FEATURE SCALING
========================
Normalizes all features to same scale
Uses StandardScaler: (x - mean) / std

Why? Different features have different ranges:
- Amount: 100 to 10,000 (large range)
- Time: 0 to 23 (small range)
- merchant_type: 1 to 5 (small range)

Without scaling, Amount dominates the model!
With scaling, all features have equal importance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_split_data():
    """Load the train-test split data from Step 4"""
    try:
        X_train_balanced = pd.read_csv('X_train_balanced.csv')
        y_train_balanced = pd.read_csv('y_train_balanced.csv').squeeze()
        X_test = pd.read_csv('X_test.csv')
        y_test = pd.read_csv('y_test.csv').squeeze()
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   X_train: {X_train_balanced.shape}")
        print(f"   y_train: {y_train_balanced.shape}")
        print(f"   X_test: {X_test.shape}")
        print(f"   y_test: {y_test.shape}")
        
        return X_train_balanced, y_train_balanced, X_test, y_test
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find required CSV files!")
        print(f"   Missing: {str(e)}")
        print("   Please run: python 4_handle_imbalance.py first")
        return None, None, None, None


def scale_features(X_train, y_train, X_test, y_test):
    """
    Scale features using StandardScaler
    """
    print("\n" + "="*60)
    print("STEP 5: FEATURE SCALING")
    print("="*60)
    
    # ============================================================
    # PART 1: SHOW ORIGINAL RANGES
    # ============================================================
    print("\n1️⃣  BEFORE SCALING")
    print("-" * 60)
    
    print("\nFeature ranges (before scaling):")
    for col in X_train.columns:
        min_val = X_train[col].min()
        max_val = X_train[col].max()
        mean_val = X_train[col].mean()
        std_val = X_train[col].std()
        print(f"  {col:15s}: min={min_val:8.2f}, max={max_val:8.2f}, mean={mean_val:8.2f}, std={std_val:8.2f}")
    
    print("\n⚠️  PROBLEM:")
    print("   Amount ranges from 100-10,000 (huge range)")
    print("   Time ranges from 0-23 (small range)")
    print("   Model treats Amount as 100x more important!")
    print("   → Need to normalize to same scale")
    
    # ============================================================
    # PART 2: CREATE SCALER
    # ============================================================
    print("\n" + "-" * 60)
    print("CREATING SCALER")
    print("-" * 60)
    
    print("\n🔧 Using StandardScaler")
    print("   Formula: (x - mean) / std_dev")
    print("   Result: mean=0, std=1 for each feature")
    
    scaler = StandardScaler()
    print("✅ StandardScaler created")
    
    # ============================================================
    # PART 3: FIT SCALER ON TRAINING DATA ONLY
    # ============================================================
    print("\n" + "-" * 60)
    print("FITTING SCALER")
    print("-" * 60)
    
    print("\n⚠️  WHY FIT ONLY ON TRAINING DATA?")
    print("   - Scaler learns mean and std from training data")
    print("   - Must NOT see test data (keeps it realistic)")
    print("   - Apply same transformation to test data")
    print("   - This prevents data leakage")
    
    print("\n🔄 Fitting scaler on training data...")
    X_train_scaled = scaler.fit_transform(X_train)
    print("✅ Scaler fitted and training data transformed")
    
    print("\n🔄 Transforming test data with same scaler...")
    X_test_scaled = scaler.transform(X_test)
    print("✅ Test data transformed")
    
    # ============================================================
    # PART 4: SHOW SCALED RANGES
    # ============================================================
    print("\n2️⃣  AFTER SCALING")
    print("-" * 60)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    
    print("\nFeature ranges (after scaling):")
    for col in X_train_scaled_df.columns:
        min_val = X_train_scaled_df[col].min()
        max_val = X_train_scaled_df[col].max()
        mean_val = X_train_scaled_df[col].mean()
        std_val = X_train_scaled_df[col].std()
        print(f"  {col:15s}: min={min_val:8.2f}, max={max_val:8.2f}, mean={mean_val:8.2f}, std={std_val:8.2f}")
    
    print("\n✅ ALL FEATURES NOW HAVE:")
    print("   • Similar scales (range: -2 to +3)")
    print("   • Mean ≈ 0")
    print("   • Std ≈ 1")
    print("   • Equal importance in model")
    
    # ============================================================
    # PART 5: EXAMPLE TRANSFORMATION
    # ============================================================
    print("\n" + "-" * 60)
    print("EXAMPLE: HOW SCALING WORKS")
    print("-" * 60)
    
    print("\nOriginal first transaction:")
    print(f"  amount:          {X_train['amount'].iloc[0]:8.2f}")
    print(f"  time:            {X_train['time'].iloc[0]:8.2f}")
    print(f"  merchant_type:   {X_train['merchant_type'].iloc[0]:8.2f}")
    print(f"  card_type:       {X_train['card_type'].iloc[0]:8.2f}")
    print(f"  international:   {X_train['international'].iloc[0]:8.2f}")
    
    print("\nScaled first transaction:")
    print(f"  amount:          {X_train_scaled[0, 0]:8.2f}")
    print(f"  time:            {X_train_scaled[0, 1]:8.2f}")
    print(f"  merchant_type:   {X_train_scaled[0, 2]:8.2f}")
    print(f"  card_type:       {X_train_scaled[0, 3]:8.2f}")
    print(f"  international:   {X_train_scaled[0, 4]:8.2f}")
    
    print("\n✅ All values now on similar scale!")
    
    # ============================================================
    # PART 6: SAVE SCALED DATA & SCALER
    # ============================================================
    print("\n" + "="*60)
    print("💾 SAVING SCALED DATA & SCALER")
    print("="*60)
    
    # Convert scaled data to DataFrames
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save scaled data
    X_train_scaled_df.to_csv('X_train_scaled.csv', index=False)
    X_test_scaled_df.to_csv('X_test_scaled.csv', index=False)
    
    print("\n✅ Scaled data saved:")
    print("   • X_train_scaled.csv")
    print("   • X_test_scaled.csv")
    
    # Save scaler for later use (for predictions)
    joblib.dump(scaler, 'scaler.pkl')
    print("\n✅ Scaler saved as 'scaler.pkl'")
    print("   (Will use this for scaling new predictions)")
    
    # ============================================================
    # PART 7: SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    print("\n✅ Data Scaling Complete!")
    print(f"   Training set: {X_train_scaled.shape[0]} × {X_train_scaled.shape[1]}")
    print(f"   Test set: {X_test_scaled.shape[0]} × {X_test_scaled.shape[1]}")
    
    print("\n✅ Why scaling helped:")
    print("   • All features now equally important")
    print("   • Faster model convergence")
    print("   • Better model performance")
    print("   • Prevents one feature dominating")
    
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    """
    Run this file to scale features
    Command: python 5_feature_scaling.py
    """
    print("\n🏦 CREDIT CARD FRAUD DETECTION PROJECT")
    print("="*60)
    
    # Load split data
    X_train, y_train, X_test, y_test = load_split_data()
    
    if X_train is not None:
        # Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, y_train, X_test, y_test)
        
        print("\n" + "="*60)
        print("✅ STEP 5 COMPLETE!")
        print("="*60)
        print("\n📝 Next Step:")
        print("   Run: python 6_train_model.py")
        print("\n")
    else:
        print("\n❌ Cannot proceed without data!")