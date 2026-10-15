"""
STEP 1: DATA COLLECTION
========================
Creates a synthetic credit card fraud dataset
Saves as 'data.csv' for use in other steps
"""

import pandas as pd
import numpy as np

def create_fraud_dataset():
    """
    Generate synthetic credit card dataset
    Returns: DataFrame with 5000 transactions
    """
    print("\n" + "="*60)
    print("STEP 1: DATA COLLECTION")
    print("="*60)
    
    np.random.seed(42)  # For reproducibility
    n_samples = 5000
    
    print(f"\n📊 Creating dataset with {n_samples} transactions...")
    
    # ============================================================
    # GENUINE TRANSACTIONS (95% of data)
    # ============================================================
    print("✓ Generating genuine transactions (95%)...")
    
    n_genuine = int(n_samples * 0.95)  # 4750 genuine transactions
    
    genuine = pd.DataFrame({
        'amount': np.random.uniform(100, 5000, n_genuine),  # ₹100 to ₹5000
        'time': np.random.randint(0, 24, n_genuine),        # 0-23 hours
        'merchant_type': np.random.choice([1, 2, 3, 4, 5], n_genuine),
        'card_type': np.random.choice([0, 1, 2], n_genuine),  # 0=Debit, 1=Credit, 2=Prepaid
        'international': np.random.choice([0, 1], n_genuine, p=[0.8, 0.2]),  # 20% international
        'fraud': 0  # Label: 0 = Genuine
    })
    
    # ============================================================
    # FRAUDULENT TRANSACTIONS (5% of data)
    # ============================================================
    print("✓ Generating fraudulent transactions (5%)...")
    
    n_fraud = n_samples - n_genuine  # 250 fraudulent transactions
    
    fraud = pd.DataFrame({
        'amount': np.random.uniform(2000, 10000, n_fraud),      # Fraud uses higher amounts
        'time': np.random.choice([0, 1, 2, 3, 23], n_fraud),    # Fraud happens late night
        'merchant_type': np.random.choice([1, 3], n_fraud),     # Fraud at specific merchants
        'card_type': np.random.choice([0, 1], n_fraud),
        'international': np.random.choice([0, 1], n_fraud, p=[0.2, 0.8]),  # 80% international
        'fraud': 1  # Label: 1 = Fraudulent
    })
    
    # ============================================================
    # COMBINE AND SHUFFLE
    # ============================================================
    print("✓ Combining genuine and fraudulent data...")
    
    df = pd.concat([genuine, fraud], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data
    
    # ============================================================
    # DATASET SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("📊 DATASET SUMMARY")
    print("="*60)
    print(f"\nTotal Transactions: {len(df)}")
    print(f"Features: {list(df.columns)}")
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nClass Distribution:")
    print(df['fraud'].value_counts())
    print(f"\nClass Distribution (%):")
    print(df['fraud'].value_counts(normalize=True) * 100)
    
    # ============================================================
    # SAVE DATASET
    # ============================================================
    print("\n" + "="*60)
    print("💾 SAVING DATASET")
    print("="*60)
    
    df.to_csv('data.csv', index=False)
    print("\n✅ Dataset saved as 'data.csv'")
    print(f"📁 File size: {len(df)} rows × {len(df.columns)} columns")
    
    return df


if __name__ == "__main__":
    """
    Run this file to generate the dataset
    Command: python 1_data_collection.py
    """
    print("\n🏦 CREDIT CARD FRAUD DETECTION PROJECT")
    print("="*60)
    
    # Create dataset
    df = create_fraud_dataset()
    
    print("\n" + "="*60)
    print("✅ STEP 1 COMPLETE!")
    print("="*60)
    print("\n📝 Next Step:")
    print("   Run: python 2_preprocessing.py")
    print("\n")