"""
STEP 2: DATA PREPROCESSING
===========================
Cleans the dataset:
- Remove duplicates
- Handle missing values
- Check data quality
- Prepare for next steps
"""

import pandas as pd
import numpy as np

def load_data():
    """Load the dataset created in Step 1"""
    try:
        df = pd.read_csv('data.csv')
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
        return df
    except FileNotFoundError:
        print("\n❌ ERROR: data.csv not found!")
        print("   Please run: python 1_data_collection.py first")
        return None


def preprocess_data(df):
    """
    Clean and prepare the dataset
    """
    print("\n" + "="*60)
    print("STEP 2: DATA PREPROCESSING")
    print("="*60)
    
    print("\n📊 BEFORE PREPROCESSING:")
    print(f"   Shape: {df.shape}")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    
    # ============================================================
    # 1. CHECK FOR MISSING VALUES
    # ============================================================
    print("\n1️⃣  CHECKING FOR MISSING VALUES")
    print("-" * 60)
    
    missing_count = df.isnull().sum()
    print(f"\nMissing values per column:")
    print(missing_count)
    
    if missing_count.sum() == 0:
        print("✅ No missing values found!")
    else:
        print("⚠️  Found missing values. Filling with mean...")
        # Fill missing values with column mean (for numerical columns)
        df = df.fillna(df.mean(numeric_only=True))
        print("✅ Missing values filled!")
    
    # ============================================================
    # 2. REMOVE DUPLICATES
    # ============================================================
    print("\n2️⃣  CHECKING FOR DUPLICATES")
    print("-" * 60)
    
    before_duplicates = len(df)
    df = df.drop_duplicates()
    after_duplicates = len(df)
    removed = before_duplicates - after_duplicates
    
    print(f"\nBefore removing duplicates: {before_duplicates}")
    print(f"After removing duplicates: {after_duplicates}")
    print(f"Duplicates removed: {removed}")
    
    if removed == 0:
        print("✅ No duplicates found!")
    else:
        print(f"✅ Removed {removed} duplicate rows!")
    
    # ============================================================
    # 3. CHECK DATA TYPES
    # ============================================================
    print("\n3️⃣  DATA TYPES")
    print("-" * 60)
    
    print("\nData types:")
    print(df.dtypes)
    
    # ============================================================
    # 4. BASIC STATISTICS
    # ============================================================
    print("\n4️⃣  BASIC STATISTICS")
    print("-" * 60)
    
    print("\nNumeric columns summary:")
    print(df.describe())
    
    # ============================================================
    # 5. CHECK FOR INVALID VALUES
    # ============================================================
    print("\n5️⃣  CHECKING FOR INVALID VALUES")
    print("-" * 60)
    
    # Amount should be positive
    invalid_amount = (df['amount'] < 0).sum()
    print(f"Negative amounts: {invalid_amount}")
    
    # Time should be 0-23
    invalid_time = ((df['time'] < 0) | (df['time'] > 23)).sum()
    print(f"Invalid times (not 0-23): {invalid_time}")
    
    # fraud should be 0 or 1
    invalid_fraud = (~df['fraud'].isin([0, 1])).sum()
    print(f"Invalid fraud labels: {invalid_fraud}")
    
    if (invalid_amount + invalid_time + invalid_fraud) == 0:
        print("✅ All values are valid!")
    
    # ============================================================
    # 6. CLASS DISTRIBUTION
    # ============================================================
    print("\n6️⃣  CLASS DISTRIBUTION (IMPORTANT!)")
    print("-" * 60)
    
    print("\nFraud vs Genuine counts:")
    fraud_counts = df['fraud'].value_counts()
    print(fraud_counts)
    
    print("\nFraud vs Genuine percentages:")
    fraud_pct = df['fraud'].value_counts(normalize=True) * 100
    print(fraud_pct)
    
    genuine_count = (df['fraud'] == 0).sum()
    fraud_count = (df['fraud'] == 1).sum()
    
    print(f"\n⚠️  CLASS IMBALANCE:")
    print(f"   Genuine: {genuine_count} ({genuine_count/len(df)*100:.1f}%)")
    print(f"   Fraud:   {fraud_count} ({fraud_count/len(df)*100:.1f}%)")
    print(f"   Imbalance ratio: {genuine_count/fraud_count:.1f}:1")
    print("   (This will be fixed in Step 4 with SMOTE)")
    
    # ============================================================
    # 7. SAVE CLEANED DATA
    # ============================================================
    print("\n" + "="*60)
    print("💾 SAVING CLEANED DATA")
    print("="*60)
    
    print("\nAFTER PREPROCESSING:")
    print(f"   Shape: {df.shape}")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    
    df.to_csv('data_cleaned.csv', index=False)
    print("\n✅ Cleaned dataset saved as 'data_cleaned.csv'")
    
    return df


if __name__ == "__main__":
    """
    Run this file to preprocess the dataset
    Command: python 2_preprocessing.py
    """
    print("\n🏦 CREDIT CARD FRAUD DETECTION PROJECT")
    print("="*60)
    
    # Load data from Step 1
    df = load_data()
    
    if df is not None:
        # Preprocess data
        df_cleaned = preprocess_data(df)
        
        print("\n" + "="*60)
        print("✅ STEP 2 COMPLETE!")
        print("="*60)
        print("\n📝 Next Step:")
        print("   Run: python 3_eda.py")
        print("\n")
    else:
        print("\n❌ Cannot proceed without data!")