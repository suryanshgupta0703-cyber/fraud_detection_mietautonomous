"""
STEP 4: HANDLE IMBALANCED DATA (SMOTE)
=====================================
⭐ MOST IMPORTANT STEP ⭐

Problem: Dataset is 95% genuine, 5% fraud
If model just predicts "all genuine", it gets 95% accuracy but catches 0% fraud!

Solution: SMOTE (Synthetic Minority Over-sampling)
- Creates synthetic fraud samples
- Balances the dataset
- Ensures model learns fraud patterns

This MUST be done BEFORE training!
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data():
    """Load the cleaned dataset from Step 3"""
    try:
        df = pd.read_csv('data_cleaned.csv')
        print(f"\n✅ Cleaned dataset loaded successfully!")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
        return df
    except FileNotFoundError:
        print("\n❌ ERROR: data_cleaned.csv not found!")
        print("   Please run: python 3_eda.py first")
        return None


def handle_imbalanced_data(df):
    """
    Handle class imbalance using SMOTE
    """
    print("\n" + "="*60)
    print("STEP 4: HANDLING IMBALANCED DATA (SMOTE)")
    print("="*60)
    
    # ============================================================
    # PART 1: EXPLAIN THE PROBLEM
    # ============================================================
    print("\n⚠️  THE PROBLEM: CLASS IMBALANCE")
    print("-" * 60)
    
    print("\n❌ If we train WITHOUT balancing:")
    print("   Model could just predict 'all genuine' for everything")
    print("   Accuracy: 95% ✓ (looks good)")
    print("   But Fraud caught: 0% ✗ (useless!)")
    
    print("\n✅ With SMOTE balancing:")
    print("   Model learns actual fraud patterns")
    print("   Catches both genuine AND fraud transactions")
    
    # ============================================================
    # PART 2: PREPARE DATA
    # ============================================================
    print("\n" + "-" * 60)
    print("PREPARING DATA")
    print("-" * 60)
    
    # Separate features and labels
    X = df.drop('fraud', axis=1)  # Features (all columns except 'fraud')
    y = df['fraud']               # Labels (fraud column)
    
    print(f"\n✓ Features (X): {X.shape}")
    print(f"  Columns: {list(X.columns)}")
    print(f"✓ Labels (y): {y.shape}")
    
    # Train-Test Split BEFORE applying SMOTE
    print("\n" + "-" * 60)
    print("TRAIN-TEST SPLIT")
    print("-" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,           # 80% train, 20% test
        random_state=42,         # For reproducibility
        stratify=y               # Keep same fraud ratio in both sets
    )
    
    print(f"\n✓ Training set: {X_train.shape[0]} samples")
    print(f"  Genuine: {(y_train == 0).sum()}")
    print(f"  Fraud:   {(y_train == 1).sum()}")
    
    print(f"\n✓ Testing set: {X_test.shape[0]} samples")
    print(f"  Genuine: {(y_test == 0).sum()}")
    print(f"  Fraud:   {(y_test == 1).sum()}")
    
    # ============================================================
    # PART 3: APPLY SMOTE (ONLY on TRAINING data!)
    # ============================================================
    print("\n" + "-" * 60)
    print("APPLYING SMOTE (Synthetic Minority Over-sampling)")
    print("-" * 60)
    
    print("\n⚠️  WHY ONLY ON TRAINING DATA?")
    print("   - SMOTE creates synthetic samples")
    print("   - Must NOT touch test set (keeps it realistic)")
    print("   - Test set should be imbalanced like real data")
    
    print("\n🔄 Creating synthetic fraud samples...")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print("✅ SMOTE Applied!")
    
    # ============================================================
    # PART 4: SHOW RESULTS
    # ============================================================
    print("\n" + "="*60)
    print("📊 SMOTE RESULTS")
    print("="*60)
    
    print("\n🔴 BEFORE SMOTE (Training Set):")
    print(f"   Genuine: {(y_train == 0).sum()}")
    print(f"   Fraud:   {(y_train == 1).sum()}")
    before_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   Ratio: {before_ratio:.1f}:1 (IMBALANCED)")
    
    print("\n🟢 AFTER SMOTE (Training Set):")
    print(f"   Genuine: {(y_train_balanced == 0).sum()}")
    print(f"   Fraud:   {(y_train_balanced == 1).sum()}")
    after_ratio = (y_train_balanced == 0).sum() / (y_train_balanced == 1).sum()
    print(f"   Ratio: {after_ratio:.1f}:1 (BALANCED)")
    
    print("\n📝 Test Set (Unchanged - stays imbalanced):")
    print(f"   Genuine: {(y_test == 0).sum()}")
    print(f"   Fraud:   {(y_test == 1).sum()}")
    test_ratio = (y_test == 0).sum() / (y_test == 1).sum()
    print(f"   Ratio: {test_ratio:.1f}:1 (REALISTIC)")
    
    # ============================================================
    # PART 5: SAVE PREPARED DATA
    # ============================================================
    print("\n" + "="*60)
    print("💾 SAVING PREPARED DATA")
    print("="*60)
    
    # Create DataFrames for saving
    X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=X.columns)
    y_train_balanced_df = pd.DataFrame(y_train_balanced, columns=['fraud'])
    
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    y_test_df = pd.DataFrame(y_test.values, columns=['fraud'])
    
    # Save to CSV
    X_train_balanced_df.to_csv('X_train_balanced.csv', index=False)
    y_train_balanced_df.to_csv('y_train_balanced.csv', index=False)
    X_test_df.to_csv('X_test.csv', index=False)
    y_test_df.to_csv('y_test.csv', index=False)
    
    print("\n✅ Files saved:")
    print("   • X_train_balanced.csv (balanced training features)")
    print("   • y_train_balanced.csv (balanced training labels)")
    print("   • X_test.csv (test features)")
    print("   • y_test.csv (test labels)")
    
    # ============================================================
    # PART 6: SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    print("\n✅ What SMOTE did:")
    print(f"   Created {(y_train_balanced == 1).sum() - (y_train == 1).sum()} synthetic fraud samples")
    print(f"   Balanced training data: {(y_train_balanced == 0).sum()} genuine, {(y_train_balanced == 1).sum()} fraud")
    print(f"   Ratio changed from {before_ratio:.1f}:1 to {after_ratio:.1f}:1")
    
    print("\n✅ Why this matters:")
    print("   - Model trains on balanced data")
    print("   - Learns fraud patterns properly")
    print("   - Test data stays imbalanced (realistic)")
    print("   - Evaluation metrics will be accurate")
    
    return X_train_balanced, y_train_balanced, X_test, y_test, X.columns


if __name__ == "__main__":
    """
    Run this file to handle imbalanced data
    Command: python 4_handle_imbalance.py
    """
    print("\n🏦 CREDIT CARD FRAUD DETECTION PROJECT")
    print("="*60)
    
    # Load cleaned data
    df = load_cleaned_data()
    
    if df is not None:
        # Handle imbalance
        X_train_balanced, y_train_balanced, X_test, y_test, feature_names = handle_imbalanced_data(df)
        
        print("\n" + "="*60)
        print("✅ STEP 4 COMPLETE!")
        print("="*60)
        print("\n📝 Next Step:")
        print("   Run: python 5_feature_scaling.py")
        print("\n")
    else:
        print("\n❌ Cannot proceed without data!")